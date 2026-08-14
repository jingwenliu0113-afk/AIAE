# PROJECT_STATUS

**最後更新**：2026-08-14（MPS 診斷口徑修正：固定順序 n=1，未證因果）

---

## 目前里程碑

**里程碑 A：資料與庫存可用** — 已完成
**里程碑 C：簡化生成可用** — 已完成（BrickGPT baseline 推論 + `.ldr` 匯出）
**庫存條件訓練資料**（§9.7）— **已完成**（Instruction Format 已做；
連通已納入 gate，穩定性僅記錄未接）
**里程碑 D（硬性約束）** — 庫存層完成；**碰撞／支撐拒絕層未做**
**F-oracle baseline** — **已完成**（`data/reports/12_f_oracle.md`）
**LoRA 訓練管線煙霧測試** — **已完成**（`data/reports/13_lora_smoke.md`）。
**Workflow 里程碑 E 尚未完成**：仍缺 B／C 的庫存或替代指標比較。
val loss 下降**不是**里程碑 E 的完成證據——它只說明管線會跑、loss mask 正確。
**MPS 速度診斷** — **已完成**（`data/reports/14_mps_speed.md`）：
劣化與 **driver 端配置**成長同步；在**單一固定順序、每條件 n=1** 下，
每 10 列 `empty_cache()` 的條件**沒有出現**同樣的劣化。
這是同時發生，**不是已證實的消除**——該條件同時也是跑第二個的那一個。
正式訓練環境仍未定案。
**下一步**：先用雙向順序或獨立行程消除順序混淆，再談長程與成本；
之後才進超參數比較與 A–E；F-pipeline 另計，未開始

> 里程碑 B（既有作品推薦）尚未開始，不阻擋目前路徑。

---

## 已完成項目與驗證證據

| 項目 | 證據 |
|---|---|
| 原生 arm64 + MPS | `platform.machine()=arm64`；`torch 2.13.0`，`mps.is_available()=True` |
| StableText2Brick 載入 | train 42,604 / test 4,785 列 |
| 積木解析器 | 5,097,330 顆積木，**解析失敗 0** |
| 軸向確定 `h→x, w→y` | 該讀法出界 **0**；反過來出界 **838,563** |
| 旋轉正規化 | 14 種寫法 → **8 種**零件；`2x1` 612,169 vs `1x2` 615,295 |
| 來源資料無碰撞 | 47,389 個結構內部碰撞 **0** |
| `object_id` 無跨 split 洩漏 | 實測 **0** |
| 反事實自然供給不足 | 18,790 個多結構 object 中，僅 **1,251** 個變體用到不同零件種類 |
| Token 長度 | 每顆 10.01 token；完整序列中位數 1,064、p95 2,420 |
| 庫存引擎 | `src/inventory/engine.py`，交易回溯／256 態 bitmask，測試涵蓋 |
| **CP-SAT re-tiling** | `src/data/retile.py`，1,129 次求解，**精確覆蓋驗證失敗 0** |
| HF 授權 | Llama-3.2-1B-Instruct **已核准**；憑證不進 Git |
| BrickGPT 推論（MPS） | bf16 載入 4s，**43–50 tok/s**，600 token / 12–14s |
| 語法約束解碼 | 單一 token 假設 **實測全部成立**；600 token → 58 顆合法積木 |
| `.ldr` 匯出 | 與官方實作**逐位元組相同**（黃金向量測試）；58 條 Type-1 + 58 個 `0 STEP` |
| Split Manifest（凍結） | 物體 24,152／1,256／2,851；結構 40,489／2,115／4,785 |
| **反事實成對資料集** | 1,200／200／200 對 → **12,800 樣本**（4 種庫存變體 × 2 臂） |
| 連通判準 | **stud-only**（底板僅為錨定指標，不參與 gate） |
| 資料洩漏 | 跨 split 物體重疊 **0**（獨立 audit 重算） |
| 全量 audit | `scripts/07_audit_dataset.py` — **0 項失敗**（從 JSONL 重算，不讀 stored checks；但共用專案 parser／predicate，非獨立實作） |
| 樣本驗證 | 每筆樣本 **9 項檢查**全數通過（見下方「檢查涵蓋什麼」） |
| CP-SAT 決定性 | 多執行緒同 seed **不可重現** → 預設 `workers=1`（且更快） |
| **庫存閘控解碼（D 組）** | 24 次生成、5,832 token 逐槽審計：語法違規 **0**、型別 **0/24**、數量 **0/24**、parse **24/24**、帳目一致 **24/24**；**prompt 區塊＝gate 期初帳本 24/24** |
| 稀疏取樣（本機環境） | 遮罩整列後 MPS 越界 **0.60%**（24/4000）；受限取樣後 **0%**；CPU 兩者皆 0 |
| **Instruction Format** | 兩臂各 12,784 列；2048 涵蓋 **12,784/12,784**（max 2,044）；tokenizer revision 已釘住、與 adapter 分開設定 |
| Instruction audit | `scripts/11_audit_instruction.py` — **全量 25,568 列重新 tokenise，0 失敗**；98.7% target 用旋轉拼法；缺少的 sample ID 經核對**恰好是 2 個完整 pair**（每臂 16 列，無單一 role／variant／臂的殘缺移除） |
| 超長移除（train） | 觸發的 `inv` rows **4**（`noinv` 0）→ pair **2** → 來源樣本 **16** → instruction 列 **32**（兩臂合計）。四個數字分列於 `10_instruction.md`，不互相代用 |
| **F-oracle baseline** | 1,600 tasks／1,476 unique／200 pairs／178 幾何；驗證失敗 **0**、`INFEASIBLE` **0**；接受 1,496（93.5%），失敗 **全為 10s timeout**（104） |
| F-oracle 積木數 | 1,399 個 `OPTIMAL` **全部與參考解完全相同**；44 個多於參考者**全是 `FEASIBLE`**（搜尋被截斷，非更差的最佳解）；少於參考 **0**（不可能，參考解本身即無預算下的最小解） |
| **F-oracle 連通（分母不可混用）** | end-to-end yield **1,124/1,600＝70.25%**；成功解中的連通率 **1,124/1,496＝75.1%**；逐幾何 all-solved-and-connected yield **58/178＝32.6%**（非連通率）；逐幾何條件比例 **58/136＝42.6%** |
| **F-oracle 已證最優子集** | 只有 `OPTIMAL` 可稱最小積木解：**1,059/1,399＝75.7% 連通**，**340 個已證最優但斷開**。`FEASIBLE`（97）未證最優，不得稱 minimum |
| **LoRA 煙霧測試** | 2,000 rows／250 完整 pair；起點為**公開 BrickGPT adapter merge 進 base 後再加新 adapter**（已驗證 merge 確實改動權重）；可訓練 1,703,936／1,237,518,336（0.138%）|
| LoRA sanity（訓練前） | prompt 全 `-100`、target＋EOS 才計 loss、無截斷、僅 LoRA 可訓練、32 個 `lora_B` 梯度非零、adapter 存回讀 64/64 相同 — **0 失敗** |
| LoRA loss | val **1.1186 → 0.2626**；train window **0.9419 → 0.2324**（僅管線訊號，非泛化結論）|
| **本機 MPS 訓練速度（阻擋項）** | 2,000 rows 一個 epoch 花 **5.76 小時**；首 200 rows **2.85s/row** → 末 200 rows **37.37s/row**（**13.1×** 漸進劣化，最差 window 95s/row）。已排除列長（有打亂）。成因見 report 14 |
| **base revision 全臂共用** | `src/model_ids.py` 單一來源；A／B／D（`BrickGPT`）與 C／E（`load_finetuned`）都載入 `9213176…`，constructor 有 `base_revision` 參數並實際傳入（有測試盯住）|
| **LoRA 冷啟動載入順序** | `scripts/13_lora_coldstart.py`：實測順序 base→公開 adapter→merge→本機 adapter；正確路徑 loss **0.1037**，錯誤路徑（裸 base ＋自訓 adapter）**0.5609**（差 0.457，證明 merge 這步是實質的）；`BrickGPT(adapter=本機路徑)` 已會**明確拒絕** |
| **MPS 速度診斷（report 14）** | 200 列／兩條件，**同一行程固定順序** `continuous → empty_cache`。200/200 列 token 數相同、loss **在儲存的四位小數精度下相同**。末 window（**模型計算時間**）`continuous` **5.198s/row**、`empty_cache` **1.248s/row** |
| **短程劣化與哪些量同步** | PyTorch **追蹤值兩條件都一樣平**（`continuous` 2.347–2.416GB、`empty_cache` 2.349–2.418GB，差 0.002GB）；**driver 配置** `continuous` 衝到 **54.6GB**（41 個取樣中 **37 個超過建議上限 37.44GB**），swap 升到 **9.24GB**、**free＋inactive pages** 低到 **0.44GB**（inactive 是可回收頁，**不是「可用記憶體」**）；`empty_cache` **0/41 超標** |
| MPS 時間落在哪 | forward 佔 62.7%、backward 37.1%；forward 平均 `continuous` **2.14s** vs `empty_cache` **0.63s**（同一批列、同一順序）|
| **report 14 的限制** | 固定順序、每條件 **n=1**，**順序與條件混淆**（`empty_cache` 跑第二，機器已被前一條件用過）；第二條件的 swap／熱狀態／OS 狀態非相同起點（只有模型與 optimizer 依同 seed 重建）。**內部機制與完整因果未證明**；overhead 只有混合值，拆不回來（標為 unknown）|
| **report 14 的量測紀律（往後）** | scheduled clear 與 teardown clear 分開計數、分開計時；loss 未四捨五入落檔；逐列 sample ID 與端到端時間；`--from-json` 全部依 stored env 渲染並做內部一致性檢查 |
| 測試套件 | 公開化審查後 **550 passed、30 skipped**（離線同結果） |

### 檢查涵蓋什麼（避免「100%」被過度解讀）

每筆樣本通過的 9 項檢查是：`exact_cover`、`voxel_identical`、`within_inventory`、
`parts_canonical`、`forbidden_absent`、`forbidden_not_offered`、`connected`、
`touches_ground`、`inventory_adds_something`。

**不包含**：物理穩定性、支撐、碰撞（re-tiling 由構造保證不重疊）、
以及生成品質。「全數通過」只描述這九項。

### 反事實資料集（stud-only gate，重新生成）

| split | 對數 | 樣本 | 物體 | 嘗試 | 產出率 | 支撐率 |
|---|---:|---:|---:|---:|---:|---:|
| train | 1,200 | 9,600 | 1,083 | 8,789 | 13.7% | 29.4% |
| val | 200 | 1,600 | 139 | 1,515 | 13.2% | 31.8% |
| test | 200 | 1,600 | 174 | 1,473 | 13.6% | 29.2% |

**接受條件是 stud 耦合單一連通**。底板不是零件、不進庫存、不寫入輸出，
因此**不可用來判定連通**——只記錄為 `n_ground_components` 錨定指標。

**遍歷剔除零件**：不再隨機選一個零件失敗就丟棄來源，而是依固定 seed 逐一嘗試。
實測 train 有 **416/1,200 對需要試超過一個零件**——舊做法會丟掉這 35%。

產出率 13–14% 的瓶頸在 control 臂（sizing：control 斷裂 257 次 vs
counterfactual 斷裂 7 次）。目前的 re-tiling formulation 與連通率下降相關，
但**成因尚未完全隔離**：per-layer 38.3%、joint 33.3%（同形狀），換成聯合求解
並未補回落差，因此不能單獨歸因於逐層求解或最少積木目標。
連通性感知的 formulation 列為未來工作。

詳見 `data/reports/08_corpus_structure.md`（固定 seed 與抽樣規則）。

### CP-SAT re-tiling benchmark（150 結構 / 1,129 次求解 / seed 0）

| 丟棄零件 | 可行率 | 中位數 | p95 | max |
|---|---:|---:|---:|---:|
| `1x1` | **9%** | 0.01s | 0.21s | 0.46s |
| 其他七種 | **100%** | 0.11–0.12s | ≤1.29s | 10.35s |

- 整體可行率 88.0%，**逾時 0 次**（上限 10s）
- **可行性由「丟哪個零件」決定，不是結構大小**。`1x1` 一掉，落單奇數格就無解。
- → **反事實生成器只丟非 `1x1` 零件**，可得 100% 可行、中位數 0.12s。

### 關鍵數字

- 積木數／結構：min 6、p25 66、**中位數 92**、p75 127、p95 252、max 409
- `max_seq_len` 覆蓋率：1024 → 46.6%；**2048 → 91.2%**；4096 → 99.8%
- 零件占比：`1x2` 24.1%、`2x6` 21.6%、`1x1` 18.3%、`2x2` 11.8%

---

### BrickGPT 煙霧測試觀察（**非正式實驗結果**）

以下是 n=2 的單次抽樣，seed 0、temp 0.6、max_bricks 60，只用來確認管線能跑通並
說明硬約束層有存在必要。**不可當成 A/D/E 組的實驗數據**——正式實驗需要固定的
test prompt 集、多個 seed、統一的物理設定，見「實驗控制」。

| prompt | 積木 | 碰撞 | 連通元件 | 庫存（不受控） |
|---|---:|---:|---:|---|
| "A simple chair." | 58 | 0 | **3** | `2x2`×25、`1x2`×21… |
| "A small car." | 58 | **4** | 1 | `1x2`×16、`2x6`×13… |

**觀察**：語法遮罩是免費的，碰撞與連通不是。這與 BrickGPT 原生保留
`max_brick_rejections=500` 拒絕取樣與 `max_regenerations=100` 物理 rollback 的設計一致。

### 從官方原始碼取得、不需自己重做的東西

- **LDraw 零件對照表**已存在（`brick_library.json`）：8 種零件全有 partID。原規劃 §9.15 任務 1 免做。
- **語法 logit masking 已內建**（`PrefixConstrainedLogitsProcessor`）。庫存閘控是在其上**擴充**，不是從零寫。
- `brick_library.json` 甚至已預留 `inventory` 欄位（設為 100000）。
- 官方訓練超參數：LR **2e-3**、rank 32、alpha 16、3 epochs、batch 64、bf16。
  原規劃寫 1e-4／5e-5，差 20–40 倍。**兩者都列為候選，不預設官方值較優**——
  官方是在 8×A6000、batch 64、無庫存條件下調出來的，本專案的資料分布與批次大小都不同。

---

## 正在進行的工作

無。下一步見下。

### D 組實測（budget = `2x4`:10, `1x2`:8, `2x2`:6，8 個 seed）

每次都恰好用滿 24 顆並以 `inventory_exhausted` 終止，違規全為 0。

**prompt 與 gate 用同一份庫存**（`generate_with_inventory`，解碼前快照一次）。
先前 `scripts/05_d_arm_eval.py` 自行 new 一個 gate 卻沒帶庫存區塊，等於
「A 組 prompt ＋ 硬 gate」——這不是任何一組的設定。修正後重跑：合規數字完全不變
（違規 0、終止原因與積木數逐 case 相同），變的只有取樣出來的結構。

**過程中發現並修掉一個框架層 bug**：HF 的 `generate` 在 128k 寬的遮罩分布上取樣，
而本機環境下 `torch.multinomial` 有 **0.60%** 機率採到 support 之外
（`data/reports/06_mps_multinomial.md`，24/4000；CPU 為 0/4000）。
每顆積木 10 個 token，症狀是座標槽吐出 `2x1 (8,11, resurrection)` 這種輸出。

**這不是靠 `top_k`／`top_p` 能修的**（已試，無效）。解法是自建解碼迴圈，
先把候選縮到約 20 個再正規化取樣，`multinomial` 在 CPU 上執行。受限後越界為 0%。

> **範圍限定**：此結論僅適用於已測環境
> （macOS 26.6.1 / arm64 / torch 2.13.0 / bfloat16 / 該取樣路徑）。
> **不宣稱適用於所有 Apple Silicon、其他 torch 版本或其他取樣器。**
> 升級 torch 後應重跑 `scripts/06_mps_multinomial_repro.py`。

---

## F-oracle（已完成，`data/reports/12_f_oracle.md`）

**定位：Oracle 上界，不是可部署方法，不可當公平的端到端比較。**
形狀直接讀 test 參考 voxel，**不經檢索、不經索引、不經任何模型**——這是 Oracle 的
定義，不是資料洩漏（此處沒有訓練、沒有可被污染的模型），但也因此**永遠不可稱為
「我們的系統」**。預期讀法是 F-oracle 減 F-pipeline，而 F-pipeline 尚未實作。

**評估單位**：一個 pair ＝ 2 role × 4 庫存框架 ＝ 8 列，且兩個 role 是**同一個形狀**
的兩種鋪法。逐列平均會把每個形狀算 8 次。四種單位全部並列，headline 取
`unique_task`（最細且非重複的單位）；群組單位採「全中才算通過」。

| 單位 | n | 全部接受 | solved-and-connected yield（/n） | 條件比例（全連通｜全接受） |
|---|---:|---:|---:|---:|
| sample | 1,600 | 93.50% | 70.25% | 1,124/1,496 ＝ 75.13% |
| **unique_task** | **1,476** | **93.77%** | **70.05%** | 1,034/1,384 ＝ 74.71% |
| pair | 200 | 76.50% | 34.50% | 69/153 ＝ 45.10% |
| unique_geometry | 178 | 76.40% | 32.58% | 58/136 ＝ 42.65% |

**yield 與條件比例分母不同、不可互換**：yield 對全部單位（逾時計入分母），
條件比例只對「全部 task 都成功」的單位。連通率絕不併入接受率。

**接受條件**＝ voxel 完全一致 ∧ 零碰撞 ∧ 未超庫存 ∧ 零非法零件（四項皆由結果重算，
不採信 solver 自述）。**stud 連通不是接受條件**：模型只在精確覆蓋與零件預算下最小化
積木數，沒有連通約束；把斷開算成失敗會把 formulation 限制報成不可行，算成成功又
會宣稱它沒有的可組裝性。因此連通率**單獨報告、絕不併入接受率**。

參數：seed 0、time limit 10s、workers 1、目標＝最小積木數。

**Replay（`--from-json`）只重繪、不重解**，並且**保留當初求解時的環境原樣**——
不會把舊結果貼上今天的 Python／OR-Tools 版本。守門比對來源檔 SHA-256、
task signature（task ID＋voxel＋庫存＋參考積木數）、seed／time limit／workers，
以及 Python／OR-Tools／platform／machine；stored runs 不得有重複 task ID，
列數必須與現有 task 完全相同。任一不符即拒絕。

**Report JSON 明確區分 `solver_env`／`provenance`／`render_env`。**
本次 run 早於部分紀錄欄位，`provenance.backfilled: true`，並逐欄標示：
`source_sha256`、`task_signature`、`python`、`ortools` 為**事後依未變資料與環境回填**
（依據：來源檔 digest 仍等於 report 04 生成時所記、mtime 早於 run、1,600 筆
reference brick count 與現場重算一致、套件安裝時間均早於求解），
其餘欄位才是求解當時直接寫入。**事後推定不得寫成當時量測。**

**過程中修掉一個會使整組數字失效的 bug**：`retile` 把「預算 dict 中缺席的零件」
視為**無限供應**（對反事實產生器是正確語意），但庫存是封閉陳述，沒列出就是沒有。
未修正時 counterfactual 臂可以拿它自己被剔除的那個零件重建。修法是由庫存同時導出
`allowed` 與 `budget`，兩者不可能不一致。此 bug 由測試在產出報告前攔下。

## LoRA 煙霧測試（已完成，`data/reports/13_lora_smoke.md`）

**定位：管線測試，不是實驗。** 只驗證「可重現的微調能從正確起點端到端跑完、
loss mask 確實正確」。**不比較超參數、不挑 checkpoint、完全不讀 test split**，
前後生成僅為 smoke observation，**不宣稱庫存合規或泛化結果**。

**起點（最重要的一項）**：BrickGPT 是 adapter 而非完整模型，若直接在 Llama 上
訓練新 adapter，會丟掉公開 checkpoint 卻仍看起來像在微調。做法是
**載入 base → 套用公開 adapter → merge 進權重 → 再掛一個新的 LoRA**，
並以 q_proj/v_proj 權重指紋驗證 merge 確實改變了權重（沒變就中止）。
公開 adapter 不被修改，存檔的只有我們自己的 delta。

**預先聲明的單一設定**：`r=16、alpha=32、dropout=0.05、LR 1e-4、batch 1 ×
grad accum 8（effective 8）、max_length 2048、1 epoch、seed 0、bf16`。
**不用 4-bit**：Apple Silicon 沒有可靠的 bitsandbytes 路徑，且 §9.8 明訂 QLoRA
是記憶體選項而非必要條件。

| 項目 | 值 |
|---|---|
| 資料 | train 2,000 rows／250 pair／244 object；val 320 rows／40 pair／38 object |
| 選取 | 依 seed 打亂已排序 pair id，**整對取用**，role／variant 完全平衡（1000/1000、各 variant 500）|
| `object_id` 重疊 | **0**（train 244 vs val 38）|
| 截斷 | **0** 列（最長 1,836 token）|
| 可訓練參數 | 1,703,936／1,237,518,336 ＝ **0.138%** |
| val loss | 1.1186 → **0.2626** |
| DataLoader workers | 實測 0／2／4 為 0.014／25.9／51.5 秒 → **選 0**（資料已預先編碼在記憶體，worker 只剩 spawn 成本）|

**阻擋項：本機 MPS 訓練速度。** 2,000 rows 一個 epoch 花 **5.76 小時**，
首 200 rows **2.85s/row**、末 200 rows **37.37s/row**（**13.1×**，最差 window 95s/row，
趨勢起伏不平滑）。§9.8 問「本機是否足以當正式訓練環境」——**以此證據，現狀不行**。

**已排除**：列長（rows 有打亂，長短平均分布）。

**記憶體：已由 report 14 定位（見下）。** 先前依 RSS 1.41GB 與
`mps_allocated_gb_end` 2.34GB 判斷「不是記憶體耗盡」——**那個判讀是錯的**。
追蹤值（`current_allocated_memory`）在快與慢的情況下**完全一樣平**，
真正在長的是 **driver 配置**。

> 我在過程中一度把變慢歸因於「我自己的並行指令競爭資源」，**這個判斷是錯的**：
> 清空機器重跑後仍然出現同樣的劣化。

## MPS 速度診斷（已完成，`data/reports/14_mps_speed.md`）

200 列 × 兩個**事前聲明**的條件，`time.perf_counter()` ＋ 每個階段邊界
`torch.mps.synchronize()`。診斷**會在記憶體內跑 optimizer update**（就是真正的
訓練列成本），只是不存 checkpoint——不寫入、不修改 `lora_smoke`
（五檔 SHA-256 已核對未變）。

**兩條件做的是同一件事，但只能講到儲存精度為止**：200/200 列 token 數相同、
200/200 列 loss **在儲存的四位小數下相同**。該次 run 的 loss 是四捨五入後才落檔的，
更細的比較事後做不到；**往後的 run 已改為保存未四捨五入的 loss**。
除了時間與記憶體，**兩條件的起始狀態也不同**（見下方順序混淆）。

| | `continuous` | `empty_cache`（每 10 列） |
|---|---:|---:|
| s/row 首 window | 1.501 | 1.483 |
| s/row 末 window | **5.198** | **1.248** |
| MPS **追蹤**配置 | 2.347–2.416 GB | 2.349–2.418 GB |
| MPS **driver** 配置 | 9.77 → **54.59 GB** | 4.03 → 23.69（結束 4.27）|
| 超過建議上限 37.44GB 的取樣 | **37/41** | **0/41** |
| swap | 0.88 → **9.24 GB** | 3.20 → 2.58 |
| 最低 free＋inactive pages | **0.44 GB** | 4.08 |

`free＋inactive` 是 `vm_stat` 能加總出來的量，**inactive 是可回收頁**，
不等於「可用記憶體」，不可用日常語意去讀。

**追蹤值永遠看不到這件事**：`current_allocated_memory` 在快與慢兩種情況下
一樣平。會長的是 driver 配置。先前 report 13 依「RSS 小、追蹤值小」判斷
「不是記憶體耗盡」，**那個判讀是錯的**；上一輪把該結論限縮是對的。

**能講到什麼程度（限縮後）**：短程劣化與 **driver allocation、swap、
memory pressure 同步**；在**單一固定順序、每條件 n=1** 的條件下，
週期性 `empty_cache()` 與「劣化未出現」**同時發生**。
這是同時發生＋一次有效的緩解，**不是**已隔離的成因。

**未排除順序混淆**：兩條件在同一行程依 `continuous → empty_cache` 固定順序執行，
每條件依同 seed 重建模型與 optimizer，但**第二條件的 swap、熱狀態、OS 與行程狀態
不是相同起點**。下一步應先用雙向順序或獨立行程消除這個混淆。

**內部機制未證明**：保留快取、碎片化、unified memory 壓力、swap thrash
都與讀數相容，本輪沒有分離。也**不宣稱**這是 report 13 那 13.1× 的全部
（那是 2,000 列，這裡 200 列）。

**時間口徑**：window 與 first/last 是**模型計算時間**（`empty_cache()` 與
記憶體探測在計時區間外）；condition 層級的平均是**含 between-row overhead
的端到端**平均。兩者不可互相引用。該次 run 的 overhead 只有**混合**的一個數字
（`continuous` 1.89s、`empty_cache` 8.96s），**拆不回來**，報告標為 unknown；
往後的 run 分開記錄 scheduled `empty_cache()` 每次／總耗時、memory probe 耗時、
以及逐列與逐 window 的端到端時間。

**清除次數分兩種**：**scheduled** 是受測的介入（`continuous` 0 次、
`empty_cache` 20 次，皆由排程推得並標為 backfill）；**teardown** 是條件計時
結束後釋放模型用的家務事，**不算介入、不在任何時間數字內**。該次 run 從未計數
teardown，因此記為 unknown——報告只寫 unknown，**不宣稱它「跑了一次」**，
有沒有跑、跑幾次都不知道。未來 schema 2 有實際計數時才顯示次數。

**未納入 process restart 條件**：重啟同時重置模型與 optimizer，
速度差會與「換了新模型」混淆，需要另外設計並說明哪些狀態延續。

**Provenance 缺口**：該次 run **未在起跑時記錄實際執行的程式碼**，
且 numeric JSON **早於**最終腳本（腳本在跑完後才修正方法與命名）。
此缺口無法事後消除；資料 SHA、selection／training-order digest、
revision 等可重建者已逐項標為 backfilled 並寫明依據。往後的 run 由
`capture_provenance()` 在模型載入前記錄 HEAD、dirty、程式與資料 SHA、
**完整 `LoraConfig_.as_dict()`**、套件版本、device／dtype、phases、停止條件、
condition order、digest 與 revision；每個 condition 另記 input-order digest，
每一列記 sample ID。

**報告 JSON 標了 schema 版本**：既有 200 列 run 是 **schema 1**——沒有逐列
sample ID、沒有逐列端到端時間、沒有 overhead 拆分、loss 只有四位小數。
這些渲染成 unknown，**不用今天的常數回填**。往後的 run 是 schema 2，
`check_replayable()` 要求它們齊備。

**`--from-json` 只依 stored JSON 渲染**：標題的 row cap、phases、condition
order **與 condition 定義／差異**、slow-row 門檻與連續次數、condition 時間上限、
clear interval、window size、memory interval 全部來自 stored env，
缺任一欄即拒絕渲染。renderer **不再持有任何 condition 描述常數**——
兩臂差在哪是該次 run 量測的一部分，只能從紀錄讀。
`check_replayable()` 另外查內部一致性：condition 名稱與 run order、
列數與上限、clear 排程與次數、逐列 sample ID 的 digest，以及
**每個 condition 實際跑的 input order 是否等於模型載入前替它宣告的那一個**。

**schema 2 的 provenance 是 fail-closed 的**：HEAD、dirty flag、code SHA、
完整 `LoraConfig`、套件版本、device／dtype、phases、停止條件、condition order、
per-condition input-order digest **缺一即拒**；且必須與 env 相符，
任一項互相矛盾即拒。判斷「缺欄」用 **key presence 而非 truthiness**——
`working_tree_dirty=False` 是乾淨工作樹的合法值，用真假值判斷會把
provenance 最完整的 run 擋掉。schema 1 不受此要求，維持原路線。

**不只查有沒有這個 key，還查裡面有沒有東西**：`head` 必須是真的 revision；
`working_tree_dirty` 必須是 bool（`False` 合法、`None` 不合法——那代表根本沒人看）；
`code_sha256` 必須涵蓋 contract 列的每一個檔且都是合法 64 位 digest；
`lora_config` 必須有 contract 列的**全部**欄位且無 `None`；
`packages` 四個版本齊備；device 必須是 `mps`（CPU 上清快取是 no-op，
兩臂會變成同一臂）；dtype 必須與 `lora_config.dtype` 一致；
phases 與 condition order 必須是非空的名稱清單；停止條件：`slow_row_streak` 必須是正整數列數，兩個秒數必須是有限正數（排除 inf／nan）；
per-condition digest 涵蓋每個 condition 且為合法 digest。
**present-but-empty 才是關鍵失效模式**（`head: null`、`code_sha256: {}`、
四缺一的 packages）——那是「長得像紀錄的洞」，正是 report 13 看起來完整的原因。

**每個 schema 各有一份完整且凍結的 replay contract**：`ReplayContract`
（frozen dataclass）持有該版本的 required env、required provenance、
required condition fields，以及 provenance／condition 兩個 validator。
`CONTRACTS = {1: …, 2: …}` 是唯一來源，`SUPPORTED_SCHEMA_VERSIONS` 由它導出。

**刻意不抽共用 base**。已寫好的紀錄是一段完成的陳述，它該包含什麼在寫下的
當時就定了。若兩個版本共用同一個 tuple，日後多要求一個欄位（多一個設定、
多一個原始碼檔、多一個超參數）就會讓所有原本正確的紀錄「回溯失效」——
那和「用今天的常數重繪舊 run」是同一個錯誤，只是低一層。
必要的 code-file 與 LoRA 欄位同樣寫死在 `SCHEMA2_CODE_FILES`／
`SCHEMA2_LORA_FIELDS`，**不從 `CODE_FILES` 或 `LoraConfig_().as_dict()` 推導**。
**要新增必要欄位就加 schema 3，schema 1／2 的意義維持不變。**

**replay 先依確切版本取得單一 contract，再用它完成全部驗證**：
`resolve_contract(version)` → required env／provenance 由 contract 列出，
兩個 validator 也由 contract 指定。**全程沒有 `schema >= n`**——
「至少版本 n」是對日後 schema 意思的猜測，而較新 schema 可以用同樣的欄位名
表達不同的東西。今天共通的規則放在 `_check_provenance_common()` 與
`_check_conditions_common()`，由各版本 validator **明確呼叫**，不是繼承來的；
schema 1 的 `required_condition_fields` 是空的（它本來就沒記那些欄位）。
未知版本直接拒絕（含 `2.0`、`True` 這種靠 `==` 混進來的值）。

**MPS 不可用時直接 fail early**：`resolve_device()` 在載入 tokenizer 與資料**之前**
就擋下。CPU 上 `empty_cache()` 是明確 no-op，`empty_cache` 臂會排 20 次清除卻一次
都沒做、結果與 `continuous` 相同，最後被 replay gate 以「clear count 與排程矛盾」
拒絕——而那時機器時間已經花掉了。`--from-json` 不受影響（重繪不需要裝置）。

**冷啟動載入順序已修正。** 自訓 adapter 是套在 **merge 後**的權重上訓練的，
目錄本身看不出這件事，所以 `BrickGPT(adapter=本機路徑)` 會靜默把它套到裸 Llama 上
——照樣載入、照樣吐積木，只有數字不對。現在：
`src/training/lora.py::load_finetuned()` 強制
`base(釘住 revision) → 公開 adapter(釘住 revision) → merge → 本機 adapter`，
checkpoint 旁寫入 `brickagain_manifest.json` 記錄順序與 adapter 雜湊，
`BrickGPT(adapter=…)` 偵測到該 manifest 會直接拒絕。
實測（`scripts/13_lora_coldstart.py`）：正確路徑 loss **0.1037**、
錯誤路徑 **0.5609**，差 0.457——這步是實質的，不是形式上的。

**Provenance 有一個補不回來的缺口。** 該次 run **沒有**記錄訓練起始時的
source commit 與檔案雜湊。`de6b51e` 是**訓練後**的第一個整理 commit
（08:36:27，比 checkpoint 的 07:52:40 晚 44 分鐘）且**包含訓練後才做的修改**，
因此**不等於**產生該次 run 的程式碼，先前報告那樣寫是錯的、已更正。
這個缺口事後無法消除。往後的 run 會在**模型載入前**記錄 HEAD、working tree
是否 dirty，以及腳本／training module／instruction encoder 的 SHA-256。

**selection digest 與 training-order digest 是兩件事**：前者是「選了哪些列、
固定排列」，後者是「`torch.randperm` 之後真正餵進模型的順序」。
兩次 run 可以選到完全相同的列卻用不同順序訓練。既有 run 的 training-order
digest 是**事後以 seed 0／torch 2.13.0 重放重建的**，已標示；
torch 升級會讓這個重建失效。往後直接記錄。

## 下一個明確動作

1. **先消除順序混淆**（阻擋正式訓練）— report 14 只在**單一固定順序、
   每條件 n=1** 下觀察到「清快取的條件沒有劣化」，而那個條件同時也是跑第二個的。
   要先用**雙向順序或獨立行程**重跑，才談得上「有沒有效」。
2. **再驗證長程是否守得住，並量測它自己的成本** — 200 列不能外推到 2,000 列；
   新腳本已會分開記錄 scheduled clear 的次數與耗時。**這兩項之前不要開 A–E。**
3. **超參數比較**（§9.8 要求至少兩組）— 本輪只跑一組預先聲明的設定
   `r=16 / alpha=32 / LR 1e-4`；官方的 `LR 2e-3 / r=32` 尚未比較
4. 碰撞／支撐拒絕層（D 組剩餘部分）
5. **連通性感知的 CP-SAT formulation** — F-oracle 實測：逐 geometry 的
   **all-solved-and-connected yield 僅 58/178＝32.6%**（這是 yield，**不是連通率**：
   該幾何只要有一個 task 逾時就不算），條件比例為 58/136＝42.6%；
   已證最優子集中 **340/1,399 已證最優但斷開**。
   這是 F 軌最該先補的一塊（F-pipeline 之前）

---

## 最近一次測試指令與結果

```bash
./.venv/bin/python -m pytest tests/ -q                  # 本機審查：550 passed, 30 skipped
HF_HUB_OFFLINE=1 ./.venv/bin/python -m pytest tests/ -q # 550 passed, 30 skipped
./.venv/bin/python scripts/04_build_counterfactual.py   # 三 split 全達標
./.venv/bin/python scripts/05_d_arm_eval.py             # 0 違規；prompt＝gate 24/24
./.venv/bin/python scripts/11_audit_instruction.py      # exit 0，全量 25,568 列
./.venv/bin/python scripts/12_f_oracle.py               # F-oracle，約 46 分鐘
./.venv/bin/python scripts/12_f_oracle.py --from-json   # 只重繪報告，不重解
./.venv/bin/python scripts/13_lora_sanity.py            # LoRA 訓練前檢查，數秒
./.venv/bin/python scripts/13_lora_smoke.py             # LoRA 煙霧測試，約 5.8 小時
./.venv/bin/python scripts/13_lora_smoke.py --from-json # 只重繪報告，不重訓
./.venv/bin/python scripts/13_lora_coldstart.py         # 自訓 adapter 冷啟動順序驗證
./.venv/bin/python scripts/14_mps_speed_diagnostic.py   # MPS 速度診斷，約 19 分鐘
./.venv/bin/python scripts/14_mps_speed_diagnostic.py --from-json  # 只重繪
./.venv/bin/python scripts/06_mps_multinomial_repro.py
./.venv/bin/python scripts/07_audit_dataset.py          # exit 0, 0 項失敗
./.venv/bin/python scripts/08_corpus_structure_study.py # 語料連通／支撐研究
```

**離線可執行**（不需網路／不載入 base model）：
`test_bricks`、`test_inventory`、`test_retile`、`test_ldr`、`test_counterfactual`。
`test_decoding` 與 `test_generate_loop` 只需已快取的 tokenizer。

---

## 修改過的檔案

- `CLAUDE.md`、`PROJECT_STATUS.md`、`.gitignore`
- `src/data/bricks.py` — 解析、旋轉正規化、stud 連接、詞彙驗證
- `src/data/retile.py` — CP-SAT 再鋪排／反事實生成
- `src/inventory/engine.py` — 庫存與交易回溯
- `src/generation/brickgpt.py` — MPS 推論、自建約束解碼迴圈、輸出解析
- `src/constraints/inventory_decode.py` — 庫存閘控（D 組）
- `src/data/splits.py` — 凍結 split manifest
- `src/data/counterfactual.py` — 反事實成對生成
- `src/data/instruction.py` — Instruction Format（Example／loss mask）
- `src/generation/prompt.py` — **唯一**的 prompt builder，訓練與推論共用
- `src/eval/oracle.py` — F-oracle（形狀已知的 CP-SAT 鋪排上界）
- `src/training/lora.py` — LoRA 取樣／loss mask／起點建構（merge 公開 adapter）
- `src/model_ids.py` — base／adapter／tokenizer 名稱與三組 revision 的**單一來源**
- `src/training/diagnostics.py` — 階段計時／記憶體取樣／停止條件（不載入模型）
- `src/rendering/ldr.py` — LDraw 匯出（對齊官方實作）
- `scripts/01_eda.py`、`scripts/02_retile_benchmark.py`
- `tests/` — `test_bricks.py`、`test_inventory.py`、`test_retile.py`、
  `test_ldr.py`、`test_decoding.py`

## 尚未 Commit 的內容

工作樹乾淨；**最新 commit 以 `git log` 為準**。

| commit | 內容 |
|---|---|
| `45c625a` | parser、inventory、EDA |
| `e8c4a07` | CP-SAT re-tiling + benchmark |
| `ee35192` | BrickGPT 推論、語法解碼、LDraw 匯出 |
| `94f4e7e` | 狀態檔更新 |
| `cdd1b74` | 交接文件對齊實測 |
| `9b7641f` | 反事實成對資料 + 凍結 split |
| `8b6d7c7` | 庫存閘控解碼（D 組） |
| `37c4947` | 記帳修正、連通 gate、distractor 修復 |
| `c19bcbf` | 改回 stud-only gate、遍歷剔除零件、終止原因統一 |
| `967330f` | unique targets、三母體研究、stagger benchmark |
| `7e4a779` | 宣稱限縮至實測範圍、逐層 solver status 修正 |
| `cdc3129` | Instruction Format |
| `d84c225` | 共用 prompt builder、旋轉規則、配對 token 成本 |
| `c59bbb6` | gate 庫存餵進 prompt、旋轉規則縮短、超長 pair 整對移除、全量 token audit、tokenizer 與 adapter 來源分離 |
| `5f652df` | D-arm eval 也走 `generate_with_inventory`（prompt＝gate），並重跑 |
| `fc1e320` | 超長移除四種計數分列、local adapter 測試改成真的載入、audit 全量核對移除原因 |
| `0836495` | F-oracle baseline（含封閉庫存修正：未列出的零件＝數量 0） |
| `51d233b` | 連通統計分母正名、replay 拒絕輸入／設定漂移 |
| `b88b135` | solver 環境與 render 環境分離、provenance 標示回填 |
| `de6b51e` | LoRA 訓練管線煙霧測試（2,000 列） |
| `3f65552` | 自訓 adapter 冷啟動載入順序、manifest、provenance 補齊 |
| `a354f39` | base revision 由 `src/model_ids.py` 全臂共用、provenance 缺口據實記錄 |
| `066d155` | 短程 MPS 速度診斷（200 列 × 兩條件）|

## 實驗輸出與 Log 位置

- `data/reports/01_eda.md` / `.json` — EDA
- `data/reports/02_retile.md` / `.json` — CP-SAT benchmark
- `data/reports/04_counterfactual.md` / `.json` — 反事實資料集報告（含 SHA-256）
- `data/reports/05_d_arm.md` / `.json` — D 組 24 次生成完整紀錄
- `data/reports/06_mps_multinomial.md` / `.json` — 稀疏取樣 reproducer
- `data/reports/07_audit.md` / `.json` — 獨立全量 audit
- `data/reports/08_corpus_structure.md` / `.json` — 三母體連通／支撐研究
- `data/reports/09_stagger_ablation.md` / `.json` — stagger 同預算 operational benchmark（已否決進生產路徑）
- `data/reports/10_instruction.md` / `.json` — Instruction Format 與序列長度（含 SHA-256）
- `data/reports/11_instruction_audit.md` / `.json` — prompt／token／庫存對齊 audit
- `data/reports/12_f_oracle.md` / `.json` — F-oracle（含每筆 1,600 次求解紀錄）
- `data/reports/13_lora_sanity.json` — LoRA 訓練前 sanity（mask／梯度／存讀）
- `data/reports/13_lora_coldstart.json` — 自訓 adapter 冷啟動載入順序驗證
- `data/reports/14_mps_speed.md` / `.json` — MPS 速度診斷（逐列、逐 window、記憶體曲線）
- `data/reports/13_lora_smoke.md` / `.json` — LoRA 煙霧測試（含 loss、成本、前後生成）
- `artifacts/checkpoints/lora_smoke/` — 自訓 adapter（6.8MB，**未進 Git**）
- `data/splits/object_splits.json` — **凍結的** split manifest（4.9 MB，進 Git）
- `data/processed/counterfactual_{train,val,test}.jsonl` — 資料集（**未進 Git**）
- `artifacts/ldraw/*.ldr` — 生成結果，**已進 Git**（`.gitignore` 只排除
  `artifacts/renders/` 與 `artifacts/checkpoints/`）
- HF 快取：`~/.cache/huggingface/`

---

## 外部阻礙與需要使用者處理的事項

**目前無阻礙。** HF 授權已於 2026-08-13 解除（Llama 授權已核准；
帳號與憑證不進 Git）。

已知後續可能需要使用者處理：

- **Gurobi 學術授權**（免費）— BrickGPT 真正的物理穩定性分析需要。沒有時會退回
  connectivity-based 檢查，較不準。等做到 D/E 組硬約束時再處理。
- **真實積木照片** — 第二優先的影像辨識階段才需要，核心完成前不會用到。
