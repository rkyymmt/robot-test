# toio × ATOM Mate for toio 障害物回避 sim2real サンプル

toio コア キューブに **ATOM Mate for toio**(ToF 距離センサー VL53L0X 内蔵)を載せたロボットに、
「マット上のゴールへ、障害物を避けながら移動する」動作を強化学習で覚えさせるサンプルです。
**シミュレータで学習し、そのまま実機で動かします(sim2real)。**
学習と推論はどちらも **Google Colab** で行います。

```
 ┌──────────── Google Colab ────────────┐
 │ 01 学習: 2D シミュレータ + PPO        │   モデル (model.zip + env_config.json)
 │ 02 推論: 方策を 10Hz で実行  ─────────┼──→ Google ドライブに保存して受け渡し
 └───────────────┬──────────────────────┘
                 │ MQTT (インターネット経由)
          ┌──────┴──────┐
          │ MQTT ブローカー │
          └──────┬──────┘
                 │ Wi-Fi
 ┌───────────────┴──────────────────────┐
 │ ATOM Lite / Matrix (ESP32)            │  ToF を読み、状態を 20Hz で送信
 │  firmware/atom_toio_bridge            │  指令を toio に中継 + 安全機能
 └───────────────┬──────────────────────┘
                 │ BLE
          toio コア キューブ (Position ID でマット上の位置・向きを取得)
```

Colab のホスト型ランタイムはクラウド上にあり toio と直接 BLE 通信できないため、
ATOM を **Wi-Fi ⇔ BLE の中継役** にしています。

## 必要なもの

| もの | 用途 |
|---|---|
| toio コア キューブ | 移動ロボット本体 |
| ATOM Mate for toio | ATOM の取り付け台・電池・ToF センサー (VL53L0X) |
| ATOM Lite または ATOM Matrix | ESP32。Wi-Fi で MQTT、BLE で toio |
| toio のプレイマット | Position ID による位置取得 (トイオ・コレクション付属マット、または簡易プレイマット) |
| 障害物 | 紙コップ・箱など。**ToF センサーの高さより背が高く**、白っぽく光沢のないもの (黒・透明・鏡面は ToF が測りにくい) |
| 2.4GHz の Wi-Fi | ATOM のネット接続 |
| Google アカウント | Colab と Google ドライブ |

## ディレクトリ構成

```
toio_sim2real/          Python パッケージ (Colab から import する)
  config.py             物理定数・マット・ドメインランダマイゼーション (DR)・報酬の設定
  core.py               ★ sim と実機で共有: 観測の組み立て、行動→速度指示値、安全停止
  sim.py                2D シミュレータ (差動二輪・モーター遅れ・ToF・通信遅延)
  env.py                gymnasium 環境
  train.py              PPO (stable-baselines3) による学習
  evaluate.py / viz.py  評価・アニメーション・グラフ
  policies.py           学習済みモデルの読み込み、比較用のルールベース制御
  protocol.py           MQTT メッセージ定義 (ファームウェアと共通)
  real_robot.py         Colab 側の MQTT クライアントと推論ループ
  calibration.py        実機の遅延・速度・旋回・ToF を測って DR と比較
  virtual_robot.py      実機の代わりに MQTT で動く仮想ロボット
notebooks/
  01_train_in_sim.ipynb      Colab: シミュレータで学習・評価
  02_run_on_real_robot.ipynb Colab: キャリブレーション → 実機で推論
firmware/               ATOM 用ファームウェア (PlatformIO / Arduino IDE)
tests/                  pytest
```

## 手順

### 1. ATOM にファームウェアを書き込む

1. `firmware/atom_toio_bridge/config.h` を編集します。
   - `WIFI_SSID` / `WIFI_PASSWORD`
   - `MQTT_HOST` など。既定は公開テストブローカー `broker.emqx.io`
   - `TOPIC_PREFIX`: **必ず他人と衝突しないランダムな文字列に変更**します (例: `toio-sim2real/3f9a1c07`)。
     Python で `import secrets; print(secrets.token_hex(4))` とすると生成できます。
2. 書き込みます。
   - **PlatformIO (推奨)**: `cd firmware && pio run -t upload && pio device monitor`
     ライブラリのバージョンは `platformio.ini` で固定しています。
   - **Arduino IDE**: `firmware/atom_toio_bridge/atom_toio_bridge.ino` を開きます。
     ボードは「M5Atom」(esp32 ボードパッケージ **2.0.x**) を選び、ライブラリマネージャで次を入れます。
     M5StackToio 1.0.5 / **NimBLE-Arduino 1.4.3 (2.x は不可)** / PubSubClient / ArduinoJson 7 / Adafruit_VL53L0X
     パーティションは「Huge APP」にします。
3. ATOM を ATOM Mate に挿し、toio に取り付けます。
   **ToF センサーが toio の前方 (両輪を正転させたときの進行方向) を向くように** 載せてください。
4. toio と ATOM Mate の電源を入れます。シリアルモニタに `[ToF] OK` `[MQTT] 接続` `[toio] OK` が出れば準備完了です。
   toio の LED が **青** になります (DISARM 状態)。

| toio の LED | 状態 |
|---|---|
| 青 | DISARM (ATOM のボタンを押すと ARM) |
| 緑 | ARM・Colab からの指令で走行中 |
| 黄 | ARM・指令待ち (Colab が止まっている) |
| 赤 | ToF 安全停止中 (前進成分をカット) |
| 紫 | MQTT 未接続 |

### 2. Colab で学習する (`notebooks/01_train_in_sim.ipynb`)

1. このフォルダ (`robot-test`) を Google ドライブの `マイドライブ` 直下にアップロードします。
2. Colab で `notebooks/01_train_in_sim.ipynb` を開き、上から順に実行します (ランタイムは **CPU で十分** です)。
   - ルールベース制御でシミュレータの様子を確認します。
   - PPO で学習します。200 万ステップで、CPU ランタイムなら 1 時間前後が目安です。
     学習結果は `runs/ppo_avoid/model.zip` と `runs/ppo_avoid/env_config.json` としてドライブに保存されます。
     途中のチェックポイントも 25 万ステップごとに `runs/ppo_avoid/checkpoints/` へ保存されます。
     セッションが切れたときは `train(..., resume_from=チェックポイント)` で続きから学習できます。
   - 評価します。ルールベースとの比較、エピソードのアニメーション、通信遅延を変えたときの頑健性を確認できます。

### 3. Colab で実機を動かす (`notebooks/02_run_on_real_robot.ipynb`)

1. MQTT の設定を `config.h` と同じ値にします。
2. **キャリブレーション** を行います。
   測るのは、通信遅延、障害物なしの ToF、前方 10cm に物を置いたときの ToF、直進速度と左右差、旋回の速さと向きです。
   実測値がシミュレータの DR 範囲に入っているかが表示されます。外れていれば、範囲を広げた設定を保存して追加学習します (下記)。
3. ATOM のボタンを押して ARM し、学習した方策で走らせます。
   マット上の 4 点を巡回しながら障害物を避けます。
   Colab のセル停止ボタンで止めると、停止指令を送ってから終了します。
4. 軌跡・ToF・指令・遅延のグラフを確認します。ログはドライブに保存されます。

`USE_VIRTUAL_ROBOT = True` にすると、**実機なしで** 仮想ロボット (シミュレータ) を相手に 02 の流れを試せます。

## sim2real のための工夫

| 工夫 | 内容 |
|---|---|
| コードの共有 | 観測の組み立て・行動の変換・安全停止は `core.py` の同じ関数を、シミュレータと実機推論の両方から呼びます。学習時と推論時で入力の定義がずれる事故を防ぎます。設定 (`env_config.json`) もモデルと一緒に保存・読み込みします。 |
| 実機の数値に基づく物理 | 車輪間隔 26.6mm、速度 ≒ 2.81mm/s/指示値、不感帯 (8 未満は停止)、1dot ≒ 1.36mm などを、toio の技術仕様と toio SDK for Unity から取っています。 |
| ドメインランダマイゼーション | エピソードごとに次の値をランダムに変えます: モーター速度係数 ±15%、左右差、不感帯、モーター時定数、スリップ、通信遅延 0.08〜0.45 秒とジッタ・パケットロス、ToF のノイズ・スケール・オフセット・欠測・誤検出・視野角・取り付け角、位置のノイズ・読み取り失敗、ロボットの大きさ。`config.py` の `DomainRandomization` で範囲を変えられます。 |
| 通信遅延のモデル化 | 「センサーをサンプリングしてから、それを元にした指令が届くまで」を遅延としてモデル化しています。実機では ATOM が自分の時計で同じ量を測り (`lat`)、キャリブレーションでシミュレータの範囲と比べます。 |
| 安全機構もシミュレート | ファームウェアの「指令タイムアウト停止」と「ToF 近接時の前進カット」をシミュレータでも同じロジックで再現しています。学習時と実機で挙動が変わりません。 |
| 履歴入力と障害物メモリ | 単眼 ToF は視野が約 25° と狭く、回避中に視野から外れた障害物に横からぶつかりやすくなります。そこで直近 0.8 秒分の観測を積み重ねます。さらに、ToF の検出点を Position ID の自己位置でマット座標に記録し、周囲 8 方向の「覚えている障害物までの距離」(10 秒で忘れる) を方策に入力します。 |
| カリキュラム | 学習初期は障害物を少なく、DR を狭くしておき、徐々に本番の難しさへ近づけます。 |
| キャリブレーション → 追加学習 | 実機の測定値が DR 範囲外なら、`calibration.suggest_config()` で範囲を広げた設定を作ります。それで `train(..., resume_from=...)` により追加学習します。 |

## 座標系

toio マットと同じです。x は右向き、y は下向き (mm、マット左上が原点)、角度は x 軸正方向を 0 として時計回りが正です。
02 のキャリブレーション (`heading_error_deg`、`turn_gain` の符号) で、実機がこの想定どおりかを確認できます。

## MQTT について (セキュリティ)

既定の `broker.emqx.io` は **誰でも購読・送信できる公開テストブローカー** です。次の 3 つの対策を入れています。

- `TOPIC_PREFIX` をランダムにする
- ファームウェア側の速度上限 (`MAX_ABS_SPEED`)
- 本体ボタンによる ARM。既定では MQTT から ARM できません

それでも、トピック名を知られれば第三者が操作できます。継続的に使う場合は、認証付きのブローカーを使ってください
(HiveMQ Cloud や EMQX Cloud の無料枠など。`MQTT_USE_TLS 1`・ポート 8883・ユーザー名とパスワード)。
Colab から 1883 番ポートに出られない場合は、`transport="websockets"` を使います。
ブローカー側の WebSocket ポートを指定してください (EMQX は 8083、TLS 付きは 8084)。

## ローカル PC で試す場合

```bash
pip install -r requirements.txt pytest
python -m pytest tests -q                                   # テスト
python -m toio_sim2real train --steps 2000000 --out runs/ppo_avoid
python -m toio_sim2real eval --model runs/ppo_avoid
python -m toio_sim2real virtual --prefix toio-sim2real/xxxx  # 仮想ロボット (別ターミナル)
python -m toio_sim2real run --model runs/ppo_avoid --prefix toio-sim2real/xxxx --duration 60
```

ローカル PC では BLE も使えますが、このサンプルは ATOM の ToF を使うため、実機との通信は常に MQTT 経由です。

## うまくいかないとき

| 症状 | 確認すること |
|---|---|
| `[ToF] VL53L0X が見つかりません` | ATOM が ATOM Mate にしっかり挿さっているか。ATOM Lite / Matrix 以外 (AtomS3 など) はピンが異なります。`I2C_PIN_A/B` を変更してください。 |
| `[toio] 見つかりません` | toio の電源。ほかのアプリやスマホと接続中でないか。 |
| Colab に状態が届かない | `TOPIC_PREFIX`・ブローカーの設定が両側で一致しているか。ATOM のシリアルログで `[MQTT] 接続` が出ているか。 |
| ToF が常に 100〜300mm 程度を返す | センサーが床やマットを見ている可能性。02 のキャリブレーションで推奨値が出ます。`tof_max_range_mm` を下げて再学習してください。 |
| ぐるぐる回る・ふらつく | 遅延が DR 範囲を超えている可能性。02 の遅延測定を確認し、範囲を広げて追加学習するか、近いリージョンのブローカーを使ってください。 |
| 前進すると角度が合わない | ToF の向きと toio の前方が一致しているか。`heading_error_deg` が大きい場合は座標系の想定違いです。 |

## 参考

- toio コア キューブ技術仕様: https://toio.github.io/toio-spec/
- toio SDK for Unity (シミュレータの物理定数の出典): https://github.com/morikatron/toio-sdk-for-unity
- M5StackToio (ESP32 から toio を操作するライブラリ): https://github.com/kenichi884/M5StackToio
- ATOM Mate for toio の ToF を PlatformIO で読む例 (I2C ピン): https://qiita.com/Teach/items/9bdada744244e6174731
- stable-baselines3: https://stable-baselines3.readthedocs.io/
