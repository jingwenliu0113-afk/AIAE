# BrickAgain 最低非 UI 交付

這份文件描述的是可在正式 Mac 專案上操作的最低交付路徑。它不執行模型研究、
不建立指標、不開啟正式測試集，也不把展示結果拿去和已封存的 Phase 2 比較。
使用者已明確排除最小兩頁式 UI，因此本交付以命令列、LDraw 與 CPU 3D 幾何預覽
完成，不另做網頁或桌面介面。

> **狀態：本次公開版本已完成獨立技術審查。**
> 審查涵蓋的是交付程式、文件與離線驗證，不是新的模型或研究結果。
> 這只描述目前的公開版本，後續延伸研究不受此限。

## 一條命令：既有作品比對

下列是目前正式 train catalogue 中一件已知可組作品的可重現冒煙範例；在正式
Mac 專案可預期退出碼 **0**。它只是操作範例，不是模型成效或成功率。

<!-- exit-zero-compare -->
```bash
./.venv/bin/python scripts/27_delivery.py \
  --mode compare \
  --caption "This train features a streamlined, elongated rectangular body composed of uniformly arranged bricks. The top is flat with evenly spaced small cylindrical protrusions, providing a cohesive and structured appearance." \
  --inventory "1x2:1,2x4:1,2x6:5" \
  --top-n 1 \
  --ldr artifacts/ldraw/delivery_compare.ldr \
  --preview artifacts/renders/delivery_compare.png
```

流程是：

```text
手動庫存＋文字需求
→ 僅載入 split=train 的作品目錄
→ 確定性詞彙 Top-N
→ 逐件精確計算缺件與完成比例
→ 在 Top-N 內把庫存足夠、接觸地面且連通者排前
→ 共用 scorer 檢查
→ LDraw＋CPU 3D 幾何預覽
```

每個候選都會列出匿名 `catalog_id`、詞彙分數、所需庫存、缺件數與是否能完整
組裝。沒有可組候選時會明確回報，不會把「最相似」說成「可組」。旋轉拼法會
正規化到同一種庫存，例如 `4x1` 與 `1x4` 都是 `1x4`；兩種拼法同時輸入會拒絕，
不自行猜測該相加或覆蓋。

## 一條命令：最低 F-pipeline

<!-- exit-zero-f-pipeline -->
```bash
./.venv/bin/python scripts/27_delivery.py \
  --mode f-pipeline \
  --caption "This train features a streamlined, elongated rectangular body composed of uniformly arranged bricks. The top is flat with evenly spaced small cylindrical protrusions, providing a cohesive and structured appearance." \
  --inventory "1x2:1,2x4:1,2x6:5" \
  --top-n 1 \
  --time-limit 2 \
  --seed 0 \
  --ldr artifacts/ldraw/delivery_f_pipeline.ldr \
  --preview artifacts/renders/delivery_f_pipeline.png
```

流程是：

```text
文字需求
→ 僅從 Train Split 取回 Top-N 形狀
→ CP-SAT 依手動庫存重新鋪磚
→ 獨立重驗 exact cover／庫存／碰撞／邊界／接觸地面／連通
→ 第一個通過的結果輸出 LDraw 與預覽
```

每次嘗試分開記錄 `OPTIMAL`／`FEASIBLE`／`INFEASIBLE`／`UNKNOWN`、求解時間、
候選 placement 數、是否確實回傳鋪排，以及每一項 checker 結果。找到鋪排但不連通
與完全無解是兩種不同狀態；逾時也不會被寫成無解。

這是「詞彙檢索＋最佳化」的最低基線。詞彙特徵支援 Unicode 與中文字元切分，
但它**不是多語 embedding，也不是語意檢索成效證據**。要把它升級為研究用的
F-pipeline，仍須另行凍結查詢集、語意模型、驗收標準與評估計畫；本輪沒有做。

## Train-only 與資料邊界

- 目錄檔名必須以 `_train.jsonl` 結尾。
- 每一列都必須明示 `split=train`；只要出現一列其他 split，整份目錄即拒絕。
- 每列 `object_id` 與 `structure_id` 都必須和凍結的
  `data/splits/object_splits.json` 相符；manifest SHA-256 不同、物件不在
  train，或 structure 屬於另一物件時都 fail-closed。非 JSON object 的列也會
  受控拒絕。
- 八種庫存 variant 不會被當成八件作品；索引只收每個 pair 的
  `control/exact` 正規列，所需庫存由磚清單重新計算。
- `--exclude-object-id` 會在排名前排除同物件，供日後已授權的評估流程使用。
- 預設目錄 `data/processed/counterfactual_train.jsonl` 是私有開發資料，不在公開
  snapshot；公開 checkout 沒有它時，指令會拒絕並說明缺檔，不會改讀別的 split。
- **凍結的是 split manifest 與它的 SHA，不是 train catalogue 本身。** 每份報告會
  記下當次 catalogue SHA，讓輸出可對帳；文件不把目前的私有 processed 檔冒稱為
  已凍結契約。

## 3D 預覽與 LDraw

既有展示命令也可以直接輸出影像：

```bash
./.venv/bin/python scripts/26_showcase.py \
  --sample tower \
  --ldr artifacts/ldraw/tower.ldr \
  --preview artifacts/renders/tower.png
```

`--preview` 以 Matplotlib Agg 在 CPU 上將每塊積木畫成 3D cuboid，可輸出 PNG 或
SVG；碰撞涉及的積木會標成洋紅色。它是幾何檢視，不是寫實渲染，也沒有物理或
穩定性分析。越界、未知零件、空結構或無法解析的磚行會拒絕，不會畫成看似合法的
圖片。LDraw 仍由既有、已對齊黃金向量的 writer 產生。

`--ldr` 與 `--preview` 必須是不同路徑；若兩者 resolve 到同一檔案，
指令會在讀取目錄或寫出任何產物前拒絕，避免後一個 writer 覆寫前一份。
不支援的 preview 副檔名也會在 LDraw 寫出前拒絕。過長的 caption 只在影像標題中
折行並以省略號限制為兩行；報告中的原始 caption 保持完整。

## 展示報告的判讀

比較與 F-pipeline 都沒有跑 decoder，因此 termination 不適用。共用 showcase
報告會如實把 `termination_accepted` 與 `deterministic_core_success` 寫成 `null`／
`n/a`；交付層另列一組不讀 termination 的「單件靜態交付檢查」，其中
明列 `touches_ground` 與 `stud_only_connected`。接觸地面不是物理穩定性，
連通也不是支撐性。
這只是該件輸出的
確定性檢查，不是 `Structural Success@K`，也不是任何比例或模型指標。

退出碼：

| code | 意義 |
|---:|---|
| 0 | 找到並重驗通過一件靜態可交付結果 |
| 1 | 流程正常完成，但本次沒有可交付結果 |
| 2 | 輸入、資料邊界或不適用旗標被拒絕 |

## 已封存研究證據的交付說明

```bash
./.venv/bin/python scripts/28_delivery_evidence.py
```

這支唯讀命令先驗證 plan、results、scores 與 project-model pointer 的既有 SHA，
把 plan 與 results 當成不透明 bytes，只讀已 materialize 的 `scores.json` 聚合及
`project_model.json`。它不開 case、不執行 scorer、不重算或改寫 Phase 2。

可交付的既有結果只有：

| arm | 已封存 Core Success@4 |
|---|---:|
| B | 6／160 |
| C | 8／160 |
| D | 47／160 |
| E | 26／160 |

這些數字只適用於事前凍結的 160 cases 與該次固定執行；沒有事前凍結的推論檢定，
不宣稱顯著或可一般化。D／E 的硬庫存條件在該次執行都是 640／640、溢出量 0，
但不得事後把 D 選成正式系統，再把同一份成績當成獨立估計。

`Structural Success@K` 沒有另行 materialize，不能把 Core Success@4 改名；
`Semantic Success@K` 沒有凍結語意門檻或人工評分；因此 `Full Success@K` 也不存在。
`exact/loose/distractor/mixed` 是單次固定執行的 strata，不是事前凍結的
τ／ρ／剔除三軸 sweep。這些缺項已封口為研究限制，模型研究線已結束，不是下一輪
待跑清單。

## 最低交付覆蓋表

| 項目 | 本輪後狀態 | 邊界 |
|---|---|---|
| 手動輸入庫存 | 完成（CLI） | 無 UI；8 種零件與旋轉正規化 |
| 既有作品比對 | 完成（最低基線） | train-only 詞彙檢索，不冒稱語意模型 |
| F-oracle／F-pipeline | F-oracle 既有；F-pipeline 最低實作完成 | F-pipeline 尚未正式評估 |
| 碰撞、接地與連通檢查 | 完成 | 接地與連通都不是物理支撐或穩定性 |
| 3D 圖片／預覽 | 完成（CPU PNG／SVG） | 幾何預覽，不是寫實渲染 |
| LDraw | 完成 | 可由展示或交付命令輸出 |
| 三軸與 Success 報告 | 證據封口完成 | 缺少的研究指標如實標記「未 materialize」 |
| 最小兩頁式介面 | 使用者明確排除 | 不列入本次最低交付 |

模型線維持 `final_H2`，不再訓練、調參、多 seed 或重選模型。Phase 3C 未授權，
碰撞／連通拒絕層至今沒有正式指標。本文件與兩支命令都不得解讀為它改善了任何
Success@K。
