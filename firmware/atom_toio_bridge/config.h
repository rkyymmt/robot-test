#pragma once
// ============================================================
//  atom_toio_bridge の設定。書き換えてから書き込むこと。
//  (Wi-Fi のパスワードを含むので、このファイルを公開リポジトリに push しない)
// ============================================================

// ---- Wi-Fi (ESP32 は 2.4GHz のみ対応) ----
#define WIFI_SSID "your-ssid"
#define WIFI_PASSWORD "your-password"

// ---- MQTT ブローカー (Colab 側の MQTT_HOST などと同じ値にする) ----
#define MQTT_HOST "broker.emqx.io"
#define MQTT_PORT 1883
#define MQTT_USER ""
#define MQTT_PASSWORD ""
#define MQTT_USE_TLS 0   // 1: TLS で接続 (ポートは通常 8883)
#define MQTT_CA_CERT ""  // TLS 時のルート CA 証明書 (PEM)。空なら証明書を検証しない (テスト用)

// トピックの接頭辞。公開ブローカーは誰でも購読・送信できるので、
// 他人と衝突しないランダムな文字列に必ず変更する (例: toio-sim2real/3f9a1c07)
#define TOPIC_PREFIX "toio-sim2real/CHANGE-ME"

// ---- toio ----
// 空なら最初に見つかった toio に接続。複数台ある場合はシリアルログに出る BLE アドレスを指定
#define TOIO_ADDRESS ""

// ---- 安全設定 (EnvConfig の同名パラメータと値をそろえる) ----
#define MAX_ABS_SPEED 80    // 速度指示値の上限 (EnvConfig.max_wheel_unit)
#define SAFETY_STOP_MM 30   // ToF がこれ未満なら前進成分をカット (EnvConfig.safety_stop_mm)
#define ALLOW_REMOTE_ARM 0  // 1: MQTT からも ARM できる (公開ブローカーでは非推奨)
