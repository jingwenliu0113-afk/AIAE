# BrickAgain 影像辨識、RAG、配色與組裝

> **狀態：本次公開版本已完成獨立技術審查。**
> 審查涵蓋的是程式、文件與離線驗證，**不是新的模型成效或研究結果**；
> 這只描述目前的公開版本，後續延伸研究不受此限。
> 這份文件描述的是**新的任務與新的測試**：公開 LEGO 影像資料上的八類辨識、
> 多語檢索、確定性配色與組裝順序。它與已封存的 Phase 2 的 160 cases
> **完全無關**，本文件任何數字都不可與 Phase 2 的結果並列，也不可改名成
> `Structural`／`Semantic`／`Full Success@K`。

**先讀這段。** 這裡有真正的量測數字（凍結 test 上的單顆分類 accuracy、
macro F1、top-3，以及偵測的 precision／recall／計數誤差），但它們只描述
**本次的公開資料、這八類、這些拍攝條件**。它們不能一般化到任意一堆 LEGO，
也不是「系統可用」的證明。

---

## 勘誤（第四十九輪，獨立技術審查後）

第四十八輪的兩項敘述經複審判定為不成立，在此更正。**沒有重跑凍結 test、
沒有重訓、沒有改動 `runs/vision/eval/frozen_test.json`**；更正的是敘述本身，
以及未來的程式行為。

### 勘誤一：偵測的 AP@50 不是偵測比較，也不是八類 mAP

凍結 test 裡那個標成 `AP@50` 的數字，是**用階段二分類器的 confidence 去排序
階段一產生的框**、再對單一類別 `brick` 算出來的 average precision。
它衡量的是「那個 confidence 把階段一的提案排得多好」，
**不是定位品質，也不是八類的 mAP@50**。

- 兩欄（CV／學習模型）**共用同一個階段一**，所以框完全一樣。
  precision、recall 與計數誤差在兩欄相同不是巧合，是**構造上必然**；
  唯一會動的只有排序。因此**這裡沒有可做的偵測器比較，本文件也不再宣稱有**。
- 原本的註記寫「單一類別時 mAP@50 與 AP@50 是同一個數字」——單獨看是對的，
  和「AP@50」這個欄名擺在一起卻會被讀成八類 mAP。這個組合是錯的。

已修正的是**程式**：`detection_report()` 現在要求呼叫端宣告分數的語意
（`SCORE_OBJECTNESS`／`SCORE_CLASSIFIER_CONFIDENCE`，見 `src/vision/metrics.py`）。
宣告為 classifier confidence 時，`average_precision_50` 會是 `null`，
數字改放在 `class_agnostic_ap50_by_classifier_confidence`，
讓後來的讀者無法把熟悉的欄名撿起來當偵測 AP 用。
真正的八類 mAP@50 也補上了實作（`per_class_average_precision()`，在類別內配對），
但**這份公開偵測資料無法支援它**——它的框沒有逐顆類別——所以它照舊回報為
「無法取得」，只是現在連 mAP 也一起具名說不可得。

`runs/vision/eval/frozen_test.json` **維持原樣**，仍用舊的欄名。
讀它的時候請以本節為準。

### 勘誤二：撤回「從三組設定中依事先凍結的判準選出」

第四十八輪報告說選中的 checkpoint 是從三組設定中依事先凍結的判準選出的。
節點上確實跑過不只一組，但**只有選中那一組的產物被回傳到 Mac**；
其餘各組的 `run_summary.json` 留在節點上，從未回傳。
本輪不得傳輸資料，因此**這台機器上沒有任何產物可以支持那個跨設定的宣稱**，
該宣稱在此**撤回**，原本那張三組比較表也一併移除。

可以支持、而且現在被機器檢查的是**同一次執行之內的 epoch 選擇**：
`runs/vision/classifier/selection_record.json`
（`./.venv/bin/python scripts/32_vision_train.py --verify-selection`）
會從磁碟上的 epoch log **重新推導**選中的 epoch，
並比對權重 SHA、程式 digest、資料與 split digest、以及 summary 與 manifest
是否互相一致。它同時把撤回寫成一個欄位（`cross_configuration_selection: null`），
而不是只寫在散文裡；有人把那個欄位填回去，`--verify-selection` 會拒絕。

---

## 一、資料來源

兩份公開資料，都是 CC BY 4.0，由 Gdańsk University of Technology 的
Bridge of Knowledge 發布。**本專案不重新散布這些影像**：原始圖片只留在
`data/raw/vision/`，不進 Git，也不進公開 snapshot。

**三條邊界不一樣，不可混為一談**（第五十輪更正；先前這裡寫成「公開 snapshot
與 GPU pack 都拒絕」，對影像 pack 是錯的）：

| 邊界 | 分類影像 | 偵測影像 |
|---|---|---|
| 公開 snapshot（`scripts/17`） | 拒絕 | 拒絕 |
| 生成軌 LoRA／`final_H2` pack（`src/training/pack.py`） | 拒絕（`data/raw/**`） | 拒絕 |
| **影像 pack（`src/vision/pack.py`）** | **攜帶**——這是它存在的理由 | **拒絕** |

影像 pack **必須**帶分類影像：節點沒有任何圖片，而分類器沒辦法用 digest 擬合。
它拒絕偵測影像，因為偵測從來不做擬合，評估留在 Mac。

| | 單顆分類 | 多顆偵測 |
|---|---|---|
| 標題 | LEGO bricks for training classification network | Tagged images with LEGO bricks |
| DOI | `10.34808/rcza-jy08` | `10.34808/anq4-rn44` |
| 版本 | 1.1 | 1.1 |
| 授權 | CC BY 4.0 | CC BY 4.0 |
| 壓縮檔大小 | 6,469,779,381 bytes | 6,016,987,324 bytes |
| 壓縮檔項目數 | 620,974 | 11,714 |
| central directory SHA-256 | `1c5692b35589f531e06bcafe5e8e85f6d40e91882c68fcc69f3ee25078793392` | `8acb0992f2b338d5b5d2d42bd96212026e826efea187ee8e7856b15a63d91522` |

資料說明論文：<https://www.nature.com/articles/s41597-023-02682-2>

### 為什麼用 central directory 的 SHA 當身分

這個 mirror 用**會過期的簽章網址**提供檔案，所以網址不是任何東西的穩定名稱；
它公布的 checksum 是 512 MB 分段 digest 的合成值，不是整個檔案的 digest。
壓縮檔的 central directory 只要有任何一個成員被新增、刪除、改名或重新壓縮
就會改變，所以本專案用它的 SHA-256 當作壓縮檔的身分。

### 只取這八類，不下載整套

兩個壓縮檔各約 6 GB、共 447 類；本專案用八類。所以壓縮檔是**按 ZIP 的設計
去讀的**：先讀 end-of-central-directory，再讀 central directory，然後對想要的
成員各發一次 byte range 請求。相鄰的成員會合併成一次 range 讀取
（`src/vision/source.py` 的 `plan_spans`），把「每個檔案兩次請求」變成
「每幾 MB 一次請求」。

每個成員都以自己的 CRC-32 與宣告大小驗證；成員名稱在被當成路徑之前先檢查
（絕對路徑、`..`、磁碟機代號、反斜線一律拒絕）；宣告大小超過上限的成員在
建立解壓器之前就拒絕，解壓器本身也有硬性輸出上限。

### 八類的對照表是**推導的，不是抄的**

```text
1x1 → 3005    1x2 → 3004    1x4 → 3010    1x6 → 3009
1x8 → 3008    2x2 → 3003    2x4 → 3001    2x6 → 2456
```

這張表不是手寫的：`src/vision/classes.py` 從
`src.rendering.ldr.PART_TO_LDRAW` **計算**出來——資料集用官方 design number
命名目錄，而長方磚的 LDraw 零件檔就是那個號碼加 `.DAT`。
`check_contract()` 在推導不再成立時會大聲失敗，而不是靜默漂移。

### 類別數：447、431 還是 448

資料集說明寫 447 類，較早的論文寫 431。**本專案不引用其中任何一個當自己的
數字**，只報實際讀到的：這個壓縮檔有 **447 個 photo 類別目錄**與
**447 個 render 類別目錄**，而**兩個 447 不是同一組**——`98197` 只有 render
沒有照片，`58176` 只有照片沒有 render，所以聯集是 **448**。

### 實際取到的樣本數

**單顆分類（207,035,740 bytes，24,458 個檔案）**

| 零件 | 真實照片 | Render |
|---|---:|---:|
| 1x1 | 432 | 3,577 |
| 1x2 | 169 | 4,823 |
| 1x4 | 224 | 1,847 |
| 1x6 | 179 | 1,990 |
| 1x8 | 36 | 2,615 |
| 2x2 | 292 | 3,046 |
| 2x4 | 216 | 2,032 |
| 2x6 | 129 | 2,851 |
| **合計** | **1,677** | **22,781** |

**多顆偵測（553,735,967 bytes，598 個檔案）**

照片 219 張、render 80 張、標註檔 299 個。

> **偵測照片是抽樣的，抽掉了什麼有記錄。** 這個壓縮檔的照片共 6.0 GB，而它的
> 框**沒有逐顆類別標註**，所以取的是一份確定性的每 n 張抽樣：
> 1 顆積木的照片每 60 張取 1（取 40 / 共 2,392，丟 2,352）、
> 2–4 顆每 6 張取 1（取 74 / 共 436，丟 362）、
> 5 顆以上全取（105 / 105）。這些數字寫在 manifest 裡，不會讓截斷看起來像
> 完整覆蓋。

Manifest digest（唯讀重驗用）：

```text
classification_manifest.json   9a1c1e8539be51515e327e5db2f1ac7de033d2c03fc3880ead9a7f2dd4e3572b
detection_manifest.json        ce08b0b493631eae411086be5406ef450bf0889f8b31937b3954ceecd60c7507
```

### 不是使用者自己拍的照片

workflow §9.11／§9.12／§14.2 原本寫「自己拍攝的真實資料 Fine-tune／Test」。
**使用者沒有自行拍攝照片**，因此本輪做的是**公開真實照片的 Fine-tune／Test**。
這是已知限制，不是等價替代：本專案沒有控制拍攝條件的能力，也沒有針對自己的
零件與光線做過任何資料收集。

---

## 二、凍結的 vision split

**邊界畫在「拍攝群組」之間，不是畫在單張圖片之間。**
同一張來源照片的三十個裁切、同一天拍攝的整批、同一個 render 實例的各個姿態，
都只會落在一邊。逐張隨機切會讓近重複同時出現在 train 與 test，回來的分數
有一部分是記憶分數——看起來是好結果，而它不是。

provenance 是從檔名讀出來的，並且**讀不出來就拒絕**，沒有「未分組」這個桶，
也沒有逐張的 fallback：

| 檔名形式 | 群組 | 例 |
|---|---|---|
| `c3_4_48NF_original_3001_…jpg` | `original` 前一段的四字元 token＝**來源照片** | `3001/photo/48NF` |
| `2456_Earth Blue_1_…jpeg` | `(design, colour)`＝**render 實例**，各姿態同組 | `2456/render/Earth Blue` |
| `IMG_20201211_171151.jpg` | 拍攝日＝**一次 session** | `session/20201211` |
| `12345_flash_01.jpg` | 同一擺盤的閃燈／無閃燈 | `arrangement/12345` |

實測 provenance 覆蓋率：分類照片 1,677 / 1,677 全部分組（394 群），
偵測照片 2,933 / 2,933 全部分組（1,008 群）。

### 凍結的 digest

```text
classification_split.json   2170880040a6e7274691fca667db70f6ab8168a86c5ae81dd45ae4ef6999fbda
detection_split.json        16599cfa20fb2213b3bc73031db53ea0e85d778b78f24b9124ff6b32701d0230
```

分層是 `(零件, population)`：每個類別在真實與合成兩個 population 裡都要進到
三個 split，否則 test 可能只有某些類別的 render，那些類別的真實 per-class
recall 就沒有定義。

**分類 split（項目數）**

| | train | validation | test |
|---|---:|---:|---:|
| 真實照片 | 1,170 | 253 | 254 |
| Render | 16,038 | 3,302 | 3,441 |
| 群組 | 517 | 108 | 113 |

真實 test 每類：1x1 66、1x2 26、1x4 34、1x6 27、1x8 **5**、2x2 47、2x4 31、2x6 18。
**1x8 只有 5 張真實 test 照片**，因為整個資料集只有 36 張 1x8 照片。
這一類的每類指標必須當成很寬的區間讀，本文件不會把它寫成一個精確數字。

**偵測 split**：train 209、validation 42、test 48（其中真實照片 33 張，
render 15 張）。

> **偵測 split 被重新凍結過一次，記在這裡而不是抹掉。**
> 第一版只用一個 stratum，結果八類中有三類完全沒有進到 test render，
> 那三類的 per-class count error 會變成沒有定義。改成把 render 也依類別分層之後
> 重新凍結，被取代的檔案留在
> `data/raw/vision/detection_split.superseded-1.json`。
> **這件事發生在任何一張 test 圖片被讀取之前**，也在任何模型被訓練之前。

### Test 只開一次

`scripts/33_vision_eval.py --test` 沒有 digest 就拒絕執行，報告用 `O_EXCL`
寫出、不覆寫；要再跑一次必須給另一個目的地，兩份都留在紀錄上。
`src/vision/train.py` 的 `load_split_items` 的 `allowed_splits` **沒有預設值**，
訓練只被給過 `train` 與 `validation`。

### 兩張圖被排除，記在紀錄上

壓縮檔裡有兩張 render 是 13×18 與 14×19 像素，低於本流程的 16 像素下限。
**這兩張被排除並記在 `run_summary.json` 的 `excluded_unreadable`**，兩張都在
train split，validation 與 test 不受影響。為兩張圖（共 24,458 張）讓整個
八個 epoch 的訓練失敗是錯的失敗方式；不說它們被排除也是錯的。

---

## 三、傳統 CV baseline

三個量測，全部可稽核，沒有任何學習或擬合：

1. **輪廓的長寬比**——foreground component 的二階矩，比軸對齊 bounding box 好，
   因為後者會隨旋轉角度變大。
2. **stud 週期**——把 gradient magnitude 投影到積木自己的主軸，得到的 profile
   的週期就是 stud 間距，用它把長度與寬度換算成 stud 數。搜尋範圍的下界是
   `1 / MAX_EXTENT`（八類中最長邊是 8 stud，所以比八分之一更短的週期不是
   stud 間距），並且**偏好基頻而不是諧波**——autocorrelation 在兩倍週期上幾乎
   一樣強，取最大值會把 1x8 讀成 1x4。
3. **stud 亮斑計數**——**權重是 0**。這不是漏寫：在 *validation* split 上掃過
   四個常數，任何非零權重都讓兩個 population 都變差 3–8 個百分點。原因在資料
   而不在程式：這個壓縮檔大量的成員是從下方或近乎側面拍攝／算繪的，亮斑同樣
   可能是管孔或鏡面邊緣。這個計數仍然量測並隨結果輸出，只是沒有投票權。

### 期望值是**推導的**，包含積木自己的高度

量測顯示 1xN 的輪廓長寬比系統性偏小：1x6 量到約 4.0 而不是 6.0。原因是幾何的
——1x6 寬 1 stud 但**高 1.2 stud**，從任何非正上方的角度看，輪廓的短邊由高度
而不是寬度決定。2xN 寬 2 stud，高度不影響它，所以 2x6 量到 3.1，正是預期值。

所以期望的輪廓長寬比是 `max(b, H) / max(a, H)`，其中 `H = LDU_BRICK / LDU_STUD
= 24 / 20 = 1.2`，**從已釘住的 LDraw 常數推導**，不是寫死的數字，也不是對資料
擬合出來的。加入這一項把 validation 的 forced top-1 從 0.254／0.271 提升到
0.333／0.333。

### 它成立的條件，以及不成立時會怎樣

這個方法要的是「重複結構看得見、背景單純」的積木。**這個公開壓縮檔大部分不是
那樣**：很多成員從下方算繪（看到中空的管）、有些近乎側面（長軸週期被壓縮或
看不見）。這是這個方法在這份資料上的真實限制，會如實報成限制，而且每個決定
所依據的特徵都隨 `Prediction.features` 一起輸出，所以 confusion matrix 裡的
任何一格都可以回推到造成它的量測。

---

## 四、學習模型

| | 值 |
|---|---|
| Backbone | `microsoft/resnet-18` |
| Revision | `65a5785d9156231087c481e0c7dd33a5ff6f7e3e` |
| 授權 | **Apache-2.0** |
| 參數量 | 11.7 M |
| 前處理 | 短邊 256 → 中心裁切 224；mean `(0.485, 0.456, 0.406)`、std `(0.229, 0.224, 0.225)` |

**為什麼不是 MobileNetV2。** `google/mobilenet_v2_1.0_224` 更輕，但它的
model card 的授權欄只寫 `other`。契約要求授權相容並記錄授權，所以選了授權
明確的 ResNet-18，它也在 workflow §9.11 的候選清單裡。

**前處理是自己寫的，並且說明為什麼。** resize 寫在
`src/vision/preprocess.resize_rgb`（半像素中心的雙線性），因為同一組算術必須在
擬合 head 的 CUDA 節點與提供推論的 Mac 上跑出同樣結果，而兩個版本的影像庫
不保證這件事。**已公布的設定要的是 bicubic，本專案用 bilinear**，這個偏差在
擬合與推論時一致套用，所以它是模型的一部分而不是模型裡的錯配——但這確實表示
這個 head 是對 bilinear 輸入擬合的。`check_processor_config` 會把 mean、std、
crop 與 crop_pct 對照已公布的設定，不符就拒絕。

### 針對 RTX 5070 Ti 的調整

節點自己回報的規格（不是憑記憶）：

```text
NVIDIA GeForce RTX 5070 Ti   capability 12.0 (sm_120)   70 SM   16,302 MiB
bf16 supported: True         cuDNN 9.19.0                torch 2.11.0+cu130
CPU: AMD Ryzen 5 7600, 6 core / 12 thread          RAM 30 GiB
```

依這些回報值（而不是卡的型號）決定的設定，全部寫進 checkpoint manifest：

| 設定 | 值 | 為什麼 |
|---|---|---|
| autocast | `bfloat16` | 問 `torch.cuda.is_bf16_supported()`，Blackwell 的 tensor core 原生支援 |
| memory format | `channels_last` | 卷積原生讀 NHWC，餵 NCHW 會讓 cuDNN 轉置每一個 activation |
| TF32 | 開 | 預設是關的 |
| cuDNN autotune | 開 | 輸入形狀固定 |
| batch size | **128** | 由回報的 VRAM 推導：≥14 GiB → 128、≥7 → 64、其他 → 32 |
| 影像載入 | 8 個 thread | 這是瓶頸而不是算術：128 張 JPEG 的解碼與縮放在 6 核上要一段時間，而卡幾毫秒就做完一步 |

**這些設定會改變算術**，所以它們被記錄而不是假設：同一台裝置、同一組設定可以
重現，和不同設定的執行**不是逐位元相同**。`--deterministic` 會把會犧牲精確性
的三項關掉。loss 與 softmax 一律在 float32 讀出——bfloat16 的驗證 loss 只有
約三位小數，而 epoch 選擇就是靠它。

### 一個被測試抓到的真實缺陷

`trainable_stages` **本來完全沒有作用**。找 backbone stage 的函式要求路徑正好
三段，而這個 backbone 的 stage 在 `resnet.encoder.stages.N`——四段。所以它什麼
都沒找到、回傳空 tuple，於是**不論設定要求解凍幾個 stage，每一次執行都只訓練
了 classifier head**（4,104 個可訓練參數）。

這種錯誤不會拋出例外：一個 linear probe 戴著 fine-tune 的標籤。修正後
`trainable_stages` 的可訓練參數是 0 → 4,104、1 → 8,397,832、2 → 10,497,544，
而且現在「要求解凍 stage 卻找不到」會**明確拒絕**，理由是 linear probe 與
fine-tune 是不同的模型、不可混為一談。`tests/test_vision_model.py` 有一組會在
舊版變紅的回歸測試。

### 執行方式

Mac 產生並驗證私人 pack，Windows WSL2 CUDA 執行，權重與 log 回傳 Mac 驗證。
`CLAUDE.md` 的「運算分流」把模型訓練指定給 Windows 節點，資料準備、凍結 split、
驗收程式與報告留在 Mac。

pack 帶：**22 個 source 檔**、**3 個腳本**（`32_vision_train.py`、
`36_vision_pack.py`、`17_public_snapshot.py`）、6 個自足測試套件、
6 份文件、兩份凍結 manifest，以及分類壓縮檔的八類成員（1,677 張照片
21.8 MB ＋ 22,781 張 render 185.2 MB）——因為**分類器沒辦法用 digest 擬合**。
pack **不**帶：任何權重（包含生成軌的 `final_H2`）、**偵測影像**、processed
文字語料、凍結的 object split、逐次執行證據、憑證，以及任何含個人絕對路徑的
文件。邊界與理由在 `src/vision/pack.py`，由 `tests/test_vision_pack.py` 逐條
測試。

> 先前這裡寫「四個腳本」是錯的，實際是三個。**第五十一輪已修好**：
> 模組說明與 build manifest 的 `carries` 都改成 three，
> 並加了一個直接數 manifest 裡 `scripts/` 條目的測試，避免再漂移。

**兩個 pack 的 source 範圍都已收窄，但收窄的程度不同，不可混為一談。**

| | 收窄前 | 收窄後 | payload 與 import closure 的關係 |
|---|---:|---:|---|
| 影像 pack | 83 個 source | **22** | **精確等於** closure（18 個模組 ＋ 4 個 package marker），逐檔列出 |
| 生成軌 LoRA／`final_H2` pack | 83 個 source | **36** | **包含** closure（29），另有 4 個 package marker 與 **3 個已揭露的生成軌模組** |

- **影像 pack 是逐檔列出的**：`PACK_SOURCE_MODULES` 就是 closure，
  `closure_problems()` 雙向核對，多一個或少一個都拒絕。
- **生成軌 pack 不是**：它保留寬白名單再逐個子樹具名拒絕（第四十九輪）。
  4 個 package marker 是 `src/data`、`src/generation`、`src/inventory`、
  `src/rendering` 的 `__init__.py`；多出的 3 個模組是
  `src/constraints/placement_decode.py`、`src/data/splits.py`、
  `src/eval/oracle.py`——**在此明確揭露，它們不在 closure 內**。
  它們與被拒的 vision／RAG／UI 不同類（都是生成軌自己的模組），
  但這仍是**已知且尚未收到最緊的地方**，留給複審決定。
  它的測試釘的是「closure 全在 payload 內」與「被拒子樹不在 payload 內」，
  **不是**「payload 等於 closure」。

兩者原本都用 `src/**/*.py` 把整個專案送上節點，包括 `src/vision/net.py`
（全專案唯一會開對外連線的模組）與 `src/ui` 的 HTTP server，而節點執行的
東西一個都沒有 import 它們。`import_closure()` 以 AST 從各自的 entry point
算出真正的相依，動態 import 逐項明列（`DYNAMIC_IMPORTS`），
未攜帶的子樹逐項具名拒絕並附理由。

> **相對 import 的解析（第五十一輪修正）。** 第五十輪的 reader 把
> `from .two import value` 裡的 `value` 當成模組去解析，整個丟掉了
> `node.module`，所以 `two.py` 會**靜默地掉出 payload**。四種形式裡有三種
> 漏掉真正的相依（`from .two import value`、`from ..top import value`、
> `from .two import *`），只有 `from . import two` 因為別名剛好就是模組而
> 僥倖通過。現在先解析 `node.module` 得到目標模組，再把「本身也是模組」的
> 別名一併納入；`*` 不當成子模組；爬過頂層仍然拒絕。
> `src/` 目前沒有任何相對 import，所以這是**潛在**缺陷而不是現行缺陷——
> 也正因如此它只能用暫存 fixture 測，而且如果沒修，下次真的寫了相對 import
> 時會在節點上才發現。修好後正式 closure 仍與 `PACK_SOURCE_MODULES` 逐項相同。

影像 pack 的測試雙向釘住：closure 內每一個檔案都必須隨包，closure 外任何
`src/**/*.py` 都不得隨包，且新出現的 `src/` 子套件若既不在 closure 也不在
拒絕表就會紅。**而且不只在 pytest 裡**：`pack_audit()` 與 `build()` 各自呼叫
`closure_problems()`，所以操作者自己跑 audit 或 build 就會看到漂移，
build 直接拒絕。

影像 pack 收窄後：**24,497 檔、221.2 MB、22 個 source**
（先前 24,558 檔、83 個 source）。
`pack_digest` 刻意不寫在這份文件裡——這份文件本身在 pack 內，
寫進去就永遠對不上；它記在 `PROJECT_STATUS.md`，
而正確的做法是自己重算：

```bash
./.venv/bin/python scripts/36_vision_pack.py --audit
./.venv/bin/python scripts/36_vision_pack.py --build "$DEST"
./.venv/bin/python scripts/36_vision_pack.py --verify "$DEST" \
    --expected-pack-digest <build 印出來的值>
```

（`$DEST` 是一個本機暫存目錄。這份文件不寫死路徑：公開快照的稽核會把
任何絕對路徑當成個人路徑擋下來，而它擋得對。）

`--verify` 不給 `--expected-pack-digest` 會**回報一個 problem**，這是設計：
拿一個東西跟它自己產生的值比對只證明算術。

> **一個實際被 verifier 抓到的問題。** macOS 的 `tar` 會為每個檔案插入
> AppleDouble（`._*`）中繼資料檔。第一次傳輸後節點上的 verify 報了
> **24,589 個「在 pack 裡但不在 manifest 裡」的檔案**——這正是那個檢查的用途：
> 到達的 pack 不是被稽核過的那個 pack。打包時要用
> `COPYFILE_DISABLE=1 tar …`。

### 訓練與選擇

> **第四十九輪撤回。** 這裡原本有一張三組設定的比較表，並宣稱選中的那組是依
> 事先凍結的判準勝出的。節點上確實跑過不只一組，但**只有選中那一組的產物被
> 回傳**，其餘各組的 summary 留在節點上、從未回傳，而本輪不得傳輸資料。
> 因此那個跨設定的宣稱在這台機器上**沒有任何產物可以查證**，已連同該表一起
> 撤回。詳見本文件開頭的〈勘誤二〉。

能夠查證的是**這一次執行之內的 epoch 選擇**，而且它是**重新推導**出來的，
不是從 manifest 上寫著的欄位抄來的：

```bash
./.venv/bin/python scripts/32_vision_train.py --selection-record   # 寫出紀錄
./.venv/bin/python scripts/32_vision_train.py --verify-selection   # 重新推導並比對
```

`runs/vision/classifier/selection_record.json` 逐項檢查八件事：權重 SHA 與
manifest 相符、權重大小相符、**選中的 epoch 等於 epoch log 上 validation loss
的最小值**、程式 digest 與這棵樹相符，以及 `run_summary.json` 與 manifest 在
權重、資料 digest、split digest 與選中 epoch 上四項一致。
只看 validation：判準是程式裡事先固定的「validation cross-entropy 最低，
完全相等時取較早的 epoch」（`src/vision/train.py` 的 `fit`）。
16 個 epoch 的 validation loss 全部留在紀錄裡，選中 **epoch 10**（0.3878）。

紀錄裡另外帶著節點回傳的一列 provenance：實際解凍的參數前綴是
`resnet.encoder.stages.2`、`resnet.encoder.stages.3` 與 `classifier`，
可訓練參數 10,497,544、凍結 683,072。這是「`trainable_stages` 這次真的生效了」
的直接證據——上一個版本的缺陷正是它完全沒有生效（見下一節）。

選中的 checkpoint：

```text
weights   sha256  5fe9d41dc5def1b7501b103b39ad40e4ad6ea03e3edd4f918886dbc301ce2e43
code      sha256  d8351d67f8e3b5e6e5d0598c3e54a2100f2617040f498685c2f2c3d035c513ba
data manifest      9a1c1e8539be51515e327e5db2f1ac7de033d2c03fc3880ead9a7f2dd4e3572b
split manifest     2170880040a6e7274691fca667db70f6ab8168a86c5ae81dd45ae4ef6999fbda
epochs 16，選中 epoch 10，trainable_stages 2，batch 128，lr 1e-3，seed 0
節點依賴：torch 2.11.0+cu130、transformers 5.15.0、numpy 2.4.4、pillow 12.2.0
```

**程式 digest 在節點與 Mac 上相同**，也就是擬合它的程式與這裡的程式逐位元組
一致。節點的 torch 是 2.11.0+cu130、Mac 是 2.13.0；這個差異記錄在 manifest 裡，
不假裝不存在。

---

## 五、凍結 test 的結果（只跑一次）

報告：`runs/vision/eval/frozen_test.json`，SHA-256
`180d6f586e5da0206e96881bd78047a137b2910b560062e1ac0697ee2ab3fa2c`。
以 `O_EXCL` 寫出，不覆寫；要再跑必須給另一個目的地，兩份都留在紀錄上。
推論在 Mac MPS 上執行。

### 單顆分類（同一份項目、同一個順序、同一份 scorer）

| | 傳統 CV | 學習模型 |
|---|---:|---:|
| **真實照片（n=254）** | | |
| accuracy（棄答算錯） | 2.36% | **94.49%** |
| forced top-1（忽略門檻） | 32.28% | 94.49% |
| coverage（答了幾成） | 6.69% | 99.61% |
| top-3 | 83.86% | 99.61% |
| macro F1 | 8.26% | 90.92% |
| 每張毫秒 | 10.1 | 10.8 |
| **Render（n=3,441）** | | |
| accuracy | 4.33% | 81.78% |
| forced top-1 | 43.68% | 82.59% |
| coverage | 9.74% | 96.45% |
| top-3 | 87.07% | 96.98% |
| macro F1 | 7.85% | 81.12% |
| 每張毫秒 | 4.8 | 6.3 |

**兩個 accuracy 欄都要讀。** 兩種方法共用一個信心門檻（0.45）——為其中一種
在比較用的資料上調門檻，會讓比較變成門檻的比較。CV 的分數分佈很散，在這個
共用規則下它大部分時候棄答，所以它的 accuracy 欄是被棄答主導的；
forced top-1 才是同一條件下的數字。**它棄答這麼多本身就是這個方法在這份資料上
的發現**，不是一個該被調掉的呈現問題。

**CV 不因為輸就被藏起來，也不因為快就少報準確率**：它每張快 0.7 毫秒，
而在真實照片上 forced top-1 是 32.28% 對 94.49%。它的 top-3 有 83.86%，
表示正確類別常常排在前三名，只是排不到第一。

**學習模型在真實照片（94.49%）比在 render（81.78%）好。** 兩個原因：
訓練取樣時真實照片被加權 6 倍；而 render 的視角變化更大（很多從下方算繪）。

**1x8 的真實 test 只有 5 張**（P 57.14%、R 80.00%）。這一類的每類指標必須
當成很寬的區間讀，不可當成精確值。

最常見的混淆（學習模型）：真實照片 `1x6→1x8` 3 次、`1x4→1x6` 2 次；
render `1x8→1x6` 73 次、`1x1→1x2` 57 次、`1x6→1x8` 49 次。
混淆集中在**長度相鄰**的類別上，這與長軸在斜視角下被壓縮一致。

### 多顆偵測與計數

> **欄名已更正，數字未動。** 下表最後一列在
> `runs/vision/eval/frozen_test.json` 裡仍寫成 `average_precision_50`。
> 那個數字是**用階段二分類器的 confidence 排序階段一的框**、對單一類別
> `brick` 算出的 average precision，**不是偵測 AP，也不是八類 mAP@50**。
> 見開頭的〈勘誤一〉。凍結檔案不重寫，也沒有重跑 test。

| | 階段二＝CV | 階段二＝學習模型 |
|---|---:|---:|
| **真實照片（28 張）** | | |
| Precision（階段一，共用） | 19.19% | 19.19% |
| Recall（階段一，共用） | 38.00% | 38.00% |
| 每張總數 MAE（階段一，共用） | 1.821 | 1.821 |
| 每張總數誤差（帶號，階段一，共用） | +1.750 | +1.750 |
| 每類 Count MAE | **無法取得** | **無法取得** |
| 八類 mAP@50 | **無法取得** | **無法取得** |
| 單類 AP，依分類器 confidence 排序（*不是*偵測 AP） | 9.48% | 10.37% |
| **Render（15 張，每張 1 顆）** | | |
| Precision（階段一，共用） | 87.50% | 87.50% |
| Recall（階段一，共用） | 93.33% | 93.33% |
| 每張總數 MAE（階段一，共用） | 0.067 | 0.067 |
| 單類 AP，依分類器 confidence 排序（*不是*偵測 AP） | 81.67% | 90.28% |

**Precision、Recall 與計數誤差在兩欄相同，這不只是「對的」，是構造上必然**：
階段一（分割）是同一段程式、同一組框，兩欄的框逐個相同。
只有框的**分數**來自階段二，所以兩欄唯一會動的就是那個排序。

**因此這裡沒有偵測器的比較可做，本文件也不宣稱有。**
兩欄最後一列的差距只說明「哪一個分類器的 confidence 更會把真的框排在前面」，
與它們找框的能力無關——它們的框根本是同一組。
固定階段一、換掉階段二確實讓**分類**成為可比的，
而分類的比較請看上一節的單顆分類表；**偵測欄不承擔那個角色**。

**帶號誤差 +1.750 表示它在真實照片上系統性地多偵測**，也就是把積木切成
好幾個框，而不是把它們併起來。這是分割在雜亂背景與反光下的失敗方式，
它出現在數字裡而不是被藏起來。

**照片的每類 Count MAE 回報為「無法取得」**，因為這份公開資料的框沒有逐顆
類別。render 的可以取得，且完整列出。

> **偵測的真實 test 是 28 張，不是 33 張。** 其中 **5 張是 6000×8000 =
> 48,000,000 像素**，超過本流程自己的 40,000,000 像素上限，所以被
> `decode_image` 拒絕並記在報告的 `unreadable` 欄裡（真實照片 truth boxes 50
> 個、predicted 99 個，都是這 28 張的）。
>
> **這個上限是本專案自己的防炸彈保護，不是資料的性質**——資料說明寫的是
> 2448×3264，而實際上有更大的。上限應該在下一次評估之前重新考慮；
> **但不會為此重跑這次的 test**：凍結計畫是「只跑一次」，而在看過兩份結果之後
> 再挑一份引用，正是凍結流程要防的事。所以這裡照實記下被排除的 5 張與原因。

---

## 六、多顆偵測是兩階段

**階段一找積木，不命名它們。** 對背景做確定性分割，每個 foreground blob 一個
框。這是單類別（`brick`）偵測器，也就按單類別評估——因為這個公開壓縮檔的框
**也是單類別的**。

**階段二命名每個框。** 每個裁切交給單顆分類器：傳統 CV 的那個，或微調過的
網路。固定階段一、換掉階段二，CV／學習模型的比較才是**分類器的比較**，
而不是兩條完整流程的比較。

這是刻意選擇而不是偷懶：公開的多顆壓縮檔把框標成 `brick`，**完全沒有逐顆類別**，
所以沒有東西可以拿來擬合一個八類偵測器；為了「有個偵測器可以訓練」而替那些框
發明類別標籤，就是發明 ground truth。兩階段路線讓每份資料只被用在它真正標註的
事情上。

因此**照片 population 的 per-class count error 會回報為「無法取得」**，並附上
理由，而不是拿一個本專案自己發明的標籤去算。同一個壓縮檔的 render **檔名裡有
design number**，所以它們單獨、完整地評估。

---

## 七、RAG：多語 embedding

| | 值 |
|---|---|
| 模型 | `intfloat/multilingual-e5-small` |
| Revision | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| 授權 | **MIT** |
| 維度 | 384 |
| 前綴 | `query: ` / `passage: `（這個家族要求，缺了會變差而且不會報錯） |
| pooling | 對 attention mask 做 mean，再 L2 normalise |
| 裝置 | **CPU**，為了可重現而不是為了速度 |

索引是**確定性的 NumPy exact cosine**，不是 FAISS 也不是 Chroma：目錄是幾千筆
384 維，一次矩陣乘法遠低於一毫秒，近似最近鄰函式庫只會多一個依賴、一個建置
步驟與一個不決定性來源。

`identity_digest` 涵蓋 repo、revision、維度、token 上限、兩個前綴與 pooling
方式。查詢時 embedder 的 digest 與索引的不符就**拒絕**，因為那些向量不可比較。
索引也記下 catalogue SHA 與凍結 split manifest SHA，任何一項不符也拒絕。

**只索引 train split**，而且用的是既有的 catalogue loader——它會拒絕任何不是
`split=train` 的列，並逐列對照凍結的 object 級 split manifest。
同物件排除是**在 search 內部**套用的，不是套在結果上，所以呼叫端不可能忘記。

**同物件排除靠的是 `object_id`，所以那份 mapping 是 fail-closed 的**
（第四十九輪修正）。舊版對 `object_rows` 用 `... or {}` 與 `.get(id, "")`：
mapping 缺席、殘缺或全是空字串都能載入成功，而空的 `object_id` **永遠不會
相等**，於是排除靜默失效——要排除的那件作品會以第一名回來，
而且沒有任何欄位說守門關掉了。現在載入時逐項拒絕：必須是 dict、
鍵與文件列**精確相等**（缺一個或多一個都拒絕）、值必須是非空字串；
`search()` 另外逐列比對 catalogue 的 `object_id`（digest 全對但 id 對錯人的
情形），並核對索引與 catalogue 的 **split manifest SHA** 是否為同一份。

**語意排名與重排位置是兩個欄位，不共用一個名字**（第四十九輪修正）。
`semantic_rank` 來自 embedding 檢索本身，`rerank_rank` 是庫存與靜態條件重排
之後的位置，只出現在重排清單裡。舊版把重排位置當成 `rank` 傳進說明產生器，
句子印的卻是「語意排名第 N」——語意第二、重排第一的作品會被說成最相似。
現在兩個都印，並註明互不推導。

**有根據的說明由結構化證據產生，沒有語言模型參與。**「可以組」這三個字只在
`fully_buildable` 為真時印出，而它同時要求庫存覆蓋每一個零件**以及**結構接地
且在相鄰層連通。這句話只在一個地方產生、只由一個條件守著；散在樣板裡總有一天
會印在一件缺四顆磚的作品旁邊。

實測一次中文查詢（不是成效指標，只是可運作的證據）：
「我想做一台小汽車」對 1,200 件 train 作品，取回語意相似度 0.82 的英文
caption 車輛，並如實回報三件候選都因為缺 `2x6` 而**不能組**。

> **本專案不宣稱 RAG 成效。** 沒有事前凍結的 retrieval test，因此沒有
> Recall@K，也沒有任何檢索品質數字。建索引本身不量測任何東西。

---

## 八、中文條件抽取

規則式，不是語言模型，而且**明確說自己是規則式的**。抽出五項：類別、最大零件
數、偏好顏色、是否允許替代、模式。

**看得懂但無法套用的條件會具名回報。** 例：

- `我要一個 50 顆的房子` → `max_parts` **不套用**，並回報
  「看到『50 顆』但沒有『以內』『最多』這類限制詞，無法判斷這是上限、下限
  還是剛好」。忽略它然後回傳一件九十顆的作品，看起來會像檢索失敗。
- `青色的東西` → 回報「青色 不在本專案的 20 色調色盤內；沒有替它挑一個最接近
  的顏色」。
- 同一句出現多種方法字眼 → `mode` 不猜，請使用者直接選。

另外有一類是「理解了但**刻意**不影響檢索」，也一併回報：類別（目錄沒有類別
欄位，用推斷出來的類別過濾就是用猜測過濾）、偏好顏色（結構軌無色，顏色交給
配色器）。

---

## 九、確定性顏色指派

形狀與顏色分開，不做「形狀×顏色」的爆炸類別。調色盤刻意**小**：20 色，每一色
都有 LDraw 標準設定的色碼與 sRGB 值。更大的表就要對色碼與 hex 做猜測，而這裡
猜錯會變成一個錯的 LDraw 檔而不是一個錯誤訊息，所以表在信心結束的地方結束。
`LDraw 115` 在表內，因為它已經是 `DEFAULT_COLOUR`——所以「不配色」與「用預設色」
產生逐位元相同的檔案。

四個性質，每一個都是拒絕而不是期望：

1. **絕不超用庫存。** 每次指派都是對既有 `Inventory` 的扣除，它在 key 空了時
   會拋出。這裡**沒有第二個計數器**，所以不可能和引擎的認知漂移。
2. **某形狀顏色湊不齊時具名拒絕。** 結構要七個 `2x4` 而庫存有四紅兩藍，答案是
   `2x4: needs 7, has 6`——不是六顆有色加一顆虛構的，也不是靜默的部分配色。
   檢查在任何一顆被上色**之前**跑完，所以拒絕不會動到庫存。
3. **確定性、可重跑。** 由下層往上、再依位置；每顆磚內先試偏好色（按給定順序）
   再依表的順序。同一結構同一庫存永遠產生同一個檔案。
4. **偏好是偏好，不是要求。** 偏好色用完就換下一個，並回報有幾顆拿到偏好色、
   幾顆沒有。
5. **一份配色，三個輸出**（第四十九輪修正；先前不是）。同一個
   `{磚索引: LDraw 色碼}` 同時餵給 LDraw writer、3D 預覽與每一張步驟圖，
   顏色值都取自這張表。部分配色一律拒絕；沒有顏色庫存時三者是同一個結構
   但**不是**同一組顏色，而且會這樣寫出來。

影像顏色辨識用 CIELAB（不是 RGB 歐氏距離）比對，信心是「最近的比第二近的近多少」
的正規化裕度——所以介於兩種灰之間的積木會回報**低信心**而不是隨便選一種灰。
選像素本身是大部分的工作：排除背景、排除最亮的鏡面高光（那是燈的反射不是塑膠）、
排除最暗的陰影。

> **一個被測試抓到的真實缺陷。** 對偵測框做顏色辨識時，原本每個裁切各自重新
> 分割。但緊貼積木的裁切幾乎全都是積木，所以以邊框估計的「背景」取樣到的是
> 積木本身，於是**紅色積木被讀成白色**。改成整張圖算一次 mask、再按框切片。

資料集的 render 檔名帶 LEGO 自己的色名（43 種）。`DATASET_COLOUR_NAMES`
**刻意只映射沒有疑義的那些**；`Bright Purple`、`Medium Nougat`、`Sand Yellow`
沒有明確對應，映射它們會把辨識分數變成猜測的分數。

---

## 十、組裝步驟

可以放下一顆磚的條件只有兩種：它落在地面，或者**已經放好的**某一顆在它下面
一層且 footprint 有交集。

這刻意**不是**全域連通性：允許先蓋好幾個接地的子結構，之後再用橫樑接起來，
這正是本專案連通性規則要允許的形狀。所以中間步驟允許是好幾塊，只有**最終**
結構要求單一元件。

**沒有任何順序可以放的磚會被回報，不會被繞過。** 一顆離地且底下整個結構都沒有
東西的磚是從上面吊著的，沒有任何順序能讓人放上它——`plan()` 會給出它的索引，
而不是編一個人做不到的順序。

**每一步都從頭重新驗證**：累積的磚清單重跑邊界、碰撞、庫存與接地檢查，不是
增量更新。會漂移的增量檢查正是這樣避開的，而一百顆磚的代價無關緊要。

LDraw 以**實際組裝順序**寫 `0 STEP`（`to_ldr_steps`）。預設寫法每顆一個
marker 的行為完全沒變，golden vector 仍然釘著它。逐步 PNG／SVG 由既有的
`write_preview` 產生，所以步驟圖與完成圖是同一段程式畫的。

**而且是同一份顏色。**（第四十九輪修正：先前不是。）配色結果以
`{磚索引: LDraw 色碼}` 同時交給 LDraw writer 與 `write_preview`，
顏色值取自同一張調色盤；每一步的前綴會**重新對映索引**，
所以第三步的磚不會拿到第一步的顏色。配色只蓋到一部分的 mapping 一律拒絕，
不會默默補成預設色——那種圖會和下載的檔案不一致而且看不出來。
沒有給顏色庫存時沒有配色：LDraw 用預設色，圖用「一種形狀一種顏色」的圖例，
頁面與序列化結果都寫明「同一個結構，不是同一組顏色」。

### 這條規則比語料庫嚴格，而這是實測到的

在 1,200 件 train 目錄作品上跑這個規則：**355 件（29.6%）可以排出合法組裝
順序**，845 件（70.4%）被拒絕，原因全部是「有積木只被上面的東西吊著」
（這種積木的中位數是 3 顆，最多 17 顆）。

這不是規則寫錯，而是**契約指定的規則比這份語料庫的內容嚴格**：契約要求
「每個非地面積木加入時，必須與已加入的相鄰下層 footprint 有交集」，
而 `src/data/bricks.py` 早就記載過語料庫的多數結構至少有一顆這種積木
（實際組裝時人會先鬆放、再用上面那顆鎖住，但那不是契約寫的規則）。

所以完整介面上，多數檢索到的既有作品**不會有組裝步驟頁**，而頁面會說出
確切理由與那些積木的索引。這是一個誠實的功能界線，不是一個 bug。
這個 29.6% 是**語料庫與規則的關係**的描述性計數，不是任何方法的成效指標。

**`stud_only_connected` 只能稱連通性。** 它是相鄰層 footprint 的 2D 交集。
不檢查質心、不檢查力矩、不檢查作品在重力下站不站得住。

---

## 十一、完整介面

```bash
./.venv/bin/python scripts/35_full_ui.py     # http://127.0.0.1:8766/
```

四頁：庫存與需求 → 照片辨識與修正 → 結果與交付 → 組裝步驟。
三個入口：RAG 既有作品、最低 F-pipeline、`final_H2` 展示。

**最小兩頁式介面完全沒有改動**，仍在 `scripts/29_ui.py`，它的 195 項測試
仍然是原本的意思。完整介面**繼承**兩頁版的 handler，所以每一個傳輸層拒絕都是
**同一段程式**而不是它的複本——這是唯一能確定沒有悄悄少掉一個的方法：

- 只綁 loopback，另外檢查 `Host` 標頭
- 結構化外部 `Origin` 在讀取本體前拒絕
- `Origin: null` 視為不透明，只能繼續走表單金鑰驗證
- 每次送出都必須帶本行程的 `csrf_token`，以 `hmac.compare_digest` 比對
- 略過本體的拒絕都送 `Connection: close`，keep-alive 不會被污染
- CSP `frame-ancestors 'none'`
- 所有數值入口只收半形、有位數與範圍上限，違反是具名 400 不是 500
- 內部錯誤只顯示一句話，traceback 只寫到終端機

**新增的只有一個 content type**：`multipart/form-data`，帶自己的界限解析器
（`src/ui/upload.py`）——本體在讀取前先依 `Content-Length` 拒絕、part 數有上限、
**未預期的欄位一律拒絕不忽略**、檔名只用於顯示且絕不當成路徑、每張圖片都走
同一個 `decode_image`（格式白名單、位元組上限、依標頭檢查像素數）。
**不寫任何暫存檔**：上傳的位元組只活在記憶體與界限化的 in-process store 裡。

### 人工修正保留三個值

每個項目分開保存 `predicted`（模型說的，永遠不被覆寫）、`edited`（人改的，
逐欄位）、`adopted`（實際採用的）。頁面、報告與庫存引擎都讀 `adopted`；
另外兩個是讓那個數字事後可以被稽核。**沒有旁路**：採用的零件標籤一律經過
`normalise_part`，所以 `4x1` 與 `1x4` 在這裡也是同一項；仍是 `unknown` 的
項目**不計入任何零件**，因為一個沒被命名的框在人說它是什麼之前不是庫存。

### `final_H2` 入口

**先驗證才載入。** `runs/project_model.json` 為每個檔案記了 digest，而驗證用的
是那個 pointer 自己的 verifier（`scripts/24_project_model.py` 的
`verify_problems`），所以這裡對「什麼是有效的 pointer」沒有第二意見。
只要有一個 digest 不符，就不會有任何解碼。

**沒有任何路徑可以重訓、調參或重選。** 硬庫存 gate 生效；placement gate
opt-in、預設關閉，開啟時頁面上會顯示「**這個閘門從未經過正式評估**，
Phase 3C 未獲授權，開啟它不能作為任何項目改善的證據，反方向也不行」。

**展示 smoke，不是評估。** 一個需求、一份庫存、一次解碼。沒有批次、沒有
Success@K、沒有 Phase 3C。實測見 `PROJECT_STATUS.md` 的當輪紀錄，那裡也記錄了
為了取得「有可交付結果」與「沒有可交付結果」兩條路徑各一個例子而各跑了幾次
單次解碼——**那些次數不構成成功率，也不可從它們算出成功率**。

沒有通過靜態交付檢查時：沒有預覽、沒有配色、沒有組裝步驟、沒有下載。
**沒有通過檢查的結構不會被畫成一張看起來合法的圖片。**

### 真實瀏覽器 smoke，以及它抓到的三個缺陷

在真實瀏覽器裡走完：中文需求 → RAG → 有根據說明；照片上傳 → 三個框 → 人工
修正（0 → 7 塊）→ 中文 RAG；`final_H2` 展示解碼 → 配色 → 3D 預覽 → LDraw →
組裝步驟 1／2／3 前後切換。**這一輪走完抓到三個只有點過去才會出現的缺陷**：

1. **表單預填了 `time_limit`**，而它只適用於 F-pipeline，所以**預設方法的第一次
   送出永遠被拒絕**。伺服器是對的，表單是錯的。
2. **`seed` 被放在 CP-SAT 專屬的 fieldset 裡**，而它同時屬於 CP-SAT 與解碼器；
   選了正式模型時整個 fieldset 被停用，**seed 就被靜默丟掉了**——正是設計明文
   禁止的靜默忽略。現在它有自己的區塊，而且對 RAG 送出 seed 會被具名拒絕。
3. **解碼上限被綁在庫存總量上**，所以上限剛好在庫存用盡的同一刻觸發，
   執行被記成 `max_bricks` 而不是硬庫存 gate 自己的 `inventory_exhausted`
   ——而後者才是 scorer 接受的終止原因。現在上限是固定值，gate 才是那個停下它的。

另外還有一個渲染缺陷（配色區塊讀了只存在於序列化形式的欄位名），因為
「有可交付結果**且**有顏色庫存」這條路徑沒有任何測試覆蓋。三個表單缺陷與這一個
都補上了會在舊版變紅的回歸測試。

---

## 十二、這裡不能證明什麼

- **不能一般化。** 所有數字只描述本次的公開資料、這八類、這些拍攝條件。
  不適用於任意一堆 LEGO。
- **不是使用者自己拍的照片。** 是公開真實照片，這是已知限制。
- **不證明 RAG 有用。** 沒有事前凍結的 retrieval test。
- **不證明 placement gate 改善任何事。** 那一層從未有過正式指標。
- **`stud_only_connected` 不是支撐，也不是穩定性。** 真正的物理穩定性分析需要
  Gurobi 學術授權，目前沒有。
- **不與 Phase 2 並列。** 這是新任務、新資料、新的凍結 split，
  與已封存的 160 cases 完全無關。
- **沒有偵測器的比較。** 兩欄共用同一個階段一，框逐個相同；
  兩欄唯一的差別是排序用的 confidence。見〈勘誤一〉。
- **沒有八類 mAP@50。** 公開偵測資料的框沒有逐顆類別。實作已備妥
  （`per_class_average_precision()`），資料不支援。
- **沒有跨設定的選擇證據。** 只有選中那一組的產物被回傳；跨設定的宣稱已撤回。
  可查證的只有同一次執行之內的 epoch 選擇（`selection_record.json`）。
- **沒有真正的字型後援保證。** 中文標題在這台機器上由 PingFang TC 畫出，
  但那是探測到的結果不是保證；一台沒有任何 CJK 字型的機器會走明確降級路徑
  （移除畫不出來的字元並在圖上說明），不會畫出豆腐方框——
  **降級本身有測試，「某台機器一定有中文字型」則沒有，也不宣稱**。

---

*本專題與任何積木製造商無關，介面不使用任何第三方商標、標誌或角色。
兩份影像資料為 CC BY 4.0，出處與作者記錄於上方與各自的 manifest；
本專案不重新散布這些影像。*
