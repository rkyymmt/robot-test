/* ----------------------------------------------------------------
  atom_toio_bridge.ino
  ATOM Mate for toio (ATOM Lite / ATOM Matrix) を Colab と toio の中継役にする。

  - toio コア キューブと BLE で接続 (M5StackToio ライブラリ)
  - ATOM Mate 内蔵の ToF センサー (VL53L0X) を連続測定
  - 状態 (ToF, Position ID, 衝突, 電池) を MQTT で 20Hz 送信
  - Colab からのモーター指令を toio に転送

  安全機能 (Colab 側のシミュレータでも同じ挙動を再現している):
  - ARM ボタン: 本体ボタンを押すまでモーターは動かない (押すたびに ARM/DISARM)
  - 指令タイムアウト: 指令が "to" ms 届かなければ停止
  - ToF 安全停止: SAFETY_STOP_MM 未満では前進成分だけカット (旋回・後退は可)
  - toio 側の自動停止: モーター指令に時間制限を付け、ATOM が固まっても止まる

  toio の LED: 青=DISARM / 緑=走行中 / 黄=指令待ち / 赤=安全停止中 / 紫=MQTT 未接続

  MQTT のメッセージ形式は toio_sim2real/protocol.py を参照。
  -------------------------------------------------------------- */
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <Wire.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Adafruit_VL53L0X.h>
#include <Toio.h>
#include <NimBLEDevice.h>

#include "config.h"

// ---- ATOM Lite / ATOM Matrix のピン ----
constexpr int BUTTON_PIN = 39;
// ATOM Mate for toio の VL53L0X は ATOM 内部 I2C (G25 / G21) につながっている。
// 資料によって SDA/SCL の記載が異なるため、起動時に両方の順序を試す
constexpr int I2C_PIN_A = 25;
constexpr int I2C_PIN_B = 21;
constexpr uint8_t VL53L0X_ADDR = 0x29;

// ---- タイミング ----
constexpr uint32_t STATE_PERIOD_MS = 50;     // 状態送信周期 (virtual_robot.py の STATE_PERIOD_S と同じ)
constexpr uint32_t MOTOR_REFRESH_MS = 100;   // 同じ指令でもこの間隔で再送して toio の自動停止を延長
constexpr uint16_t MOTOR_DURATION_MS = 250;  // toio 側の自動停止時間
constexpr uint32_t TOF_STALE_MS = 300;       // これより古い ToF 値は無効扱い
constexpr uint32_t TOIO_RETRY_MS = 3000;

// toio コア キューブの GATT
static const char* TOIO_SERVICE_UUID = "10b20100-5b3b-4571-9508-cf3efcd7bbae";
static const char* TOIO_MOTOR_UUID = "10b20102-5b3b-4571-9508-cf3efcd7bbae";

Toio toio;
ToioCore* cube = nullptr;
NimBLERemoteCharacteristic* motorChar = nullptr;
Adafruit_VL53L0X lox;
bool tofReady = false;

#if MQTT_USE_TLS
WiFiClientSecure net;
#else
WiFiClient net;
#endif
PubSubClient mqtt(net);
String topicState, topicCmd, topicStatus;

// ---- ロボットの状態 ----
struct Pose {
  uint16_t x = 0, y = 0, angle = 0;
  bool valid = false;
} pose;
int tofMm = -2;  // -1: 測定範囲外 (障害物なし), -2: センサー異常
uint32_t tofMs = 0;
bool clashLatched = false;
uint8_t battery = 0;

// ---- Colab からの指令 ----
struct Command {
  long seq = -1;
  int left = 0, right = 0;
  uint32_t timeoutMs = 500;
  uint32_t recvMs = 0;
  bool valid = false;
} cmd;
long lastLatencyMs = -1;

bool armed = false;
bool shieldActive = false;
int sentLeft = 0, sentRight = 0;
uint32_t lastMotorMs = 0;
uint32_t stateSeq = 0, lastStateMs = 0, lastToioTryMs = 0;

// ================================================================ ToF
bool probeI2C(int sda, int scl) {
  Wire.end();
  Wire.begin(sda, scl, 400000);
  Wire.beginTransmission(VL53L0X_ADDR);
  return Wire.endTransmission() == 0;
}

void setupTof() {
  bool found = probeI2C(I2C_PIN_A, I2C_PIN_B) || probeI2C(I2C_PIN_B, I2C_PIN_A);
  if (!found) {
    Serial.println("[ToF] VL53L0X が見つかりません (ATOM Mate の装着を確認)");
    return;
  }
  if (!lox.begin(VL53L0X_ADDR, false, &Wire)) {
    Serial.println("[ToF] VL53L0X の初期化に失敗");
    return;
  }
  lox.startRangeContinuous(33);
  tofReady = true;
  Serial.println("[ToF] OK");
}

void pollTof(uint32_t now) {
  if (!tofReady) return;
  if (lox.isRangeComplete()) {
    uint16_t mm = lox.readRangeResult();
    tofMm = (mm == 0xffff || mm >= 8000) ? -1 : mm;
    tofMs = now;
  }
  if (now - tofMs > TOF_STALE_MS) tofMm = -2;  // 測定が止まっている
}

// ================================================================ toio
bool connectToio() {
  Serial.println("[toio] スキャン中...");
  std::vector<ToioCore*> found = toio.scan(3);
  for (ToioCore* c : found) {
    if (strlen(TOIO_ADDRESS) > 0 && c->getAddress() != std::string(TOIO_ADDRESS)) continue;
    Serial.printf("[toio] 接続: %s (%s)\n", c->getName().c_str(), c->getAddress().c_str());
    if (!c->connect()) continue;
    cube = c;
    cube->onIDReaderData([](ToioCoreIDData d) {
      if (d.type == ToioCoreIDTypePosition) {
        pose.x = d.position.cubePosX;
        pose.y = d.position.cubePosY;
        pose.angle = d.position.cubeAngleDegree;
        pose.valid = true;
      } else {
        pose.valid = false;  // マット外・持ち上げ・カード (Standard ID)
      }
    });
    cube->onMotion([](ToioCoreMotionData m) {
      if (m.clash) clashLatched = true;
    });
    cube->onBattery([](uint8_t level) { battery = level; });
    // 最短 10ms 間隔で通知 (0xff: 変化があれば通知、静止中は 300ms 止める)。
    // 0x01 にすると角度が 3° 以上変わるまで通知されず精度が落ちる
    cube->setIDnotificationSettings(1, 0xff);
    battery = cube->getBatteryLevel();

    // モーター指令は応答なし書き込みで送る (ライブラリの controlMotor は応答待ちで遅いため)
    motorChar = nullptr;
    for (NimBLEClient* client : *NimBLEDevice::getClientList()) {
      if (!client->isConnected()) continue;
      NimBLERemoteService* svc = client->getService(TOIO_SERVICE_UUID);
      if (svc) motorChar = svc->getCharacteristic(TOIO_MOTOR_UUID);
      if (motorChar) break;
    }
    Serial.printf("[toio] OK (battery %u%%, fast motor write: %s)\n", battery, motorChar ? "yes" : "no");
    return true;
  }
  Serial.println("[toio] 見つかりません。toio の電源を入れてください");
  return false;
}

void writeMotor(int left, int right, uint16_t durationMs) {
  uint8_t data[8] = {0x02,
                     0x01, (uint8_t)(left >= 0 ? 0x01 : 0x02), (uint8_t)abs(left),
                     0x02, (uint8_t)(right >= 0 ? 0x01 : 0x02), (uint8_t)abs(right),
                     (uint8_t)(durationMs / 10)};
  if (motorChar) {
    motorChar->writeValue(data, sizeof(data), false);
  } else {
    cube->controlMotor(left >= 0, abs(left), right >= 0, abs(right), durationMs);
  }
}

// toio_sim2real/core.py の apply_safety_shield と同一ロジック
bool applySafetyShield(int& left, int& right) {
  if (tofMm >= 0 && tofMm < SAFETY_STOP_MM && left + right > 0) {
    int w = (left - right) / 2;
    left = w;
    right = -w;
    return true;
  }
  return false;
}

void updateMotors(uint32_t now) {
  if (!cube || !cube->isConnected()) return;
  int left = 0, right = 0;
  bool fresh = cmd.valid && (now - cmd.recvMs) <= cmd.timeoutMs;
  if (armed && fresh) {
    left = cmd.left;
    right = cmd.right;
  }
  shieldActive = applySafetyShield(left, right);
  bool moving = left != 0 || right != 0;
  if (left != sentLeft || right != sentRight || (moving && now - lastMotorMs >= MOTOR_REFRESH_MS)) {
    writeMotor(left, right, moving ? MOTOR_DURATION_MS : 0);
    sentLeft = left;
    sentRight = right;
    lastMotorMs = now;
  }
}

void setArmed(bool on) {
  if (on && (!cube || !cube->isConnected())) return;
  armed = on;
  cmd.valid = false;  // ARM 前に届いていた指令では動かさない
  if (cube) cube->playSoundEffect(on ? 0 : 2);
  Serial.printf("[ARM] %s\n", on ? "ARM" : "DISARM");
}

void updateLed(uint32_t now) {
  static uint32_t lastColor = 0xffffffff;
  if (!cube || !cube->isConnected()) return;
  uint32_t color;
  if (!mqtt.connected()) color = 0x800080;                                   // 紫
  else if (!armed) color = 0x0000ff;                                         // 青
  else if (shieldActive) color = 0xff0000;                                   // 赤
  else if (cmd.valid && now - cmd.recvMs <= cmd.timeoutMs) color = 0x00ff00;  // 緑
  else color = 0xffa000;                                                     // 黄
  if (color != lastColor) {
    cube->turnOnLed((color >> 16) & 0xff, (color >> 8) & 0xff, color & 0xff);
    lastColor = color;
  }
}

// ================================================================ ボタン
void pollButton(uint32_t now) {
  static bool last = HIGH;
  static uint32_t lastChange = 0;
  bool v = digitalRead(BUTTON_PIN);
  if (v != last && now - lastChange > 50) {
    lastChange = now;
    last = v;
    if (v == LOW) setArmed(!armed);
  }
}

// ================================================================ MQTT
void onMqttMessage(char* topic, byte* payload, unsigned int len) {
  JsonDocument doc;
  if (deserializeJson(doc, payload, len)) return;
  uint32_t now = millis();
  if (doc["arm"].is<int>()) {
    bool on = doc["arm"].as<int>() != 0;
    if (!on || ALLOW_REMOTE_ARM) setArmed(on);
  }
  if (doc["seq"].is<long>()) {
    int left = doc["l"] | 0;
    int right = doc["r"] | 0;
    int timeoutMs = doc["to"] | 500;
    cmd.seq = doc["seq"].as<long>();
    cmd.left = constrain(left, -MAX_ABS_SPEED, MAX_ABS_SPEED);
    cmd.right = constrain(right, -MAX_ABS_SPEED, MAX_ABS_SPEED);
    cmd.timeoutMs = constrain(timeoutMs, 100, 2000);
    cmd.recvMs = now;
    cmd.valid = true;
    // 指令の元になった状態をサンプリングしてからの経過時間 = 制御ループ遅延
    if (doc["sms"].is<uint32_t>()) lastLatencyMs = (long)(now - doc["sms"].as<uint32_t>());
  }
}

void ensureMqtt(uint32_t now) {
  static uint32_t lastTry = 0;
  if (WiFi.status() != WL_CONNECTED) {
    if (now - lastTry > 10000) {
      lastTry = now;
      Serial.println("[WiFi] 再接続...");
      WiFi.disconnect();
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    }
    return;
  }
  if (mqtt.connected() || now - lastTry < 2000) return;
  lastTry = now;
  String clientId = "atom-toio-" + String((uint32_t)ESP.getEfuseMac(), HEX);
  bool ok = strlen(MQTT_USER) > 0
                ? mqtt.connect(clientId.c_str(), MQTT_USER, MQTT_PASSWORD, topicStatus.c_str(), 0, true, "offline")
                : mqtt.connect(clientId.c_str(), topicStatus.c_str(), 0, true, "offline");
  if (ok) {
    mqtt.subscribe(topicCmd.c_str());
    mqtt.publish(topicStatus.c_str(), "online", true);
    Serial.printf("[MQTT] 接続: %s:%d  prefix=%s\n", MQTT_HOST, MQTT_PORT, TOPIC_PREFIX);
  } else {
    Serial.printf("[MQTT] 接続失敗 (state=%d)\n", mqtt.state());
  }
}

void publishState(uint32_t now) {
  if (now - lastStateMs < STATE_PERIOD_MS || !mqtt.connected()) return;
  lastStateMs = now;
  bool ble = cube && cube->isConnected();
  char buf[256];
  int n = snprintf(buf, sizeof(buf),
                   "{\"seq\":%lu,\"ms\":%lu,\"tof\":%d,\"x\":%u,\"y\":%u,\"a\":%u,\"pv\":%d,\"clash\":%d,"
                   "\"bat\":%u,\"arm\":%d,\"ack\":%ld,\"lat\":%ld,\"sh\":%d,\"l\":%d,\"r\":%d,\"ble\":%d}",
                   (unsigned long)++stateSeq, (unsigned long)now, tofMm, pose.x, pose.y, pose.angle,
                   ble && pose.valid, clashLatched, battery, armed, cmd.seq, lastLatencyMs, shieldActive,
                   sentLeft, sentRight, ble);
  mqtt.publish(topicState.c_str(), (const uint8_t*)buf, n, false);
  clashLatched = false;
}

// ================================================================ main
void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT);
  Serial.println("\n=== atom_toio_bridge ===");

  topicState = String(TOPIC_PREFIX) + "/state";
  topicCmd = String(TOPIC_PREFIX) + "/cmd";
  topicStatus = String(TOPIC_PREFIX) + "/status";

  setupTof();

  // 注意: BLE と Wi-Fi を併用するときは Wi-Fi のモデムスリープが必須なので WiFi.setSleep(false) は呼ばない
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
#if MQTT_USE_TLS
  if (strlen(MQTT_CA_CERT) > 0) net.setCACert(MQTT_CA_CERT);
  else net.setInsecure();
#endif
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMqttMessage);
  mqtt.setBufferSize(512);
  mqtt.setSocketTimeout(5);

  connectToio();
  lastToioTryMs = millis();
}

void loop() {
  uint32_t now = millis();
  ensureMqtt(now);
  mqtt.loop();
  if (!cube || !cube->isConnected()) {
    if (armed || motorChar) {  // 切断を検知したら DISARM (再接続後は再度ボタンで ARM)
      Serial.println("[toio] 切断されました");
      armed = false;
      motorChar = nullptr;
      pose.valid = false;
    }
    if (now - lastToioTryMs > TOIO_RETRY_MS) {
      connectToio();
      lastToioTryMs = millis();
    }
  } else {
    toio.loop();
  }
  now = millis();
  pollTof(now);
  pollButton(now);
  updateMotors(now);
  publishState(now);
  updateLed(now);
  delay(1);
}
