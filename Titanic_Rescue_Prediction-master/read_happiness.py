# -*- coding: utf-8 -*-
"""
讀取《2017 世界幸福報告》 world-happiness-report-2017.csv
=========================================================
在 PyCharm 直接按右鍵 Run 即可。會在終端印出資料內容與基本摘要。

欄位說明：
    Country                       : 國家／地區名稱
    Happiness.Rank                : 幸福排名（1 = 最幸福）
    Happiness.Score               : 幸福分數（0~10，越高越幸福）
    Whisker.high / Whisker.low    : 幸福分數信賴區間的上界 / 下界
    Economy..GDP.per.Capita.      : 人均 GDP（經濟）對幸福的貢獻
    Family                        : 社會支持 / 家庭的貢獻
    Health..Life.Expectancy.      : 健康預期壽命的貢獻
    Freedom                       : 自由的貢獻
    Generosity                    : 慷慨程度的貢獻
    Trust..Government.Corruption. : 對政府的信任（貪腐感受）的貢獻
    Dystopia.Residual             : 反烏托邦殘差（基準值 + 無法解釋的部分）
"""

import os
import sys

# 讓終端輸出使用 UTF-8，避免 Windows 主控台顯示中文變成亂碼
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

# ---- 基本設定 ----------------------------------------------------------
pd.set_option("display.max_columns", None)   # 印出時不省略欄位
pd.set_option("display.max_rows", None)       # 印出時不省略列（要看全部才開）
pd.set_option("display.width", 220)

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "data", "world-happiness-report-2017.csv")


def section(title: str) -> None:
    """在終端印出分隔標題。"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---- 讀取資料 ----------------------------------------------------------
# skip_blank_lines=True（預設）會自動略過檔尾的空白行
df = pd.read_csv(CSV_PATH)

section("1. 資料形狀 (rows, columns)")
print(df.shape)

section("2. 欄位名稱")
for i, col in enumerate(df.columns):
    print(f"  [{i}] {col}")

section("3. 欄位型別與非空數量 (info)")
df.info()

section("4. 前 10 筆資料")
print(df.head(10))

section("5. 後 10 筆資料")
print(df.tail(10))

section("6. 數值欄位統計摘要 (describe)")
print(df.describe())

section("7. 缺失值統計")
miss = df.isnull().sum()
if miss.sum() == 0:
    print("  沒有任何缺失值 ✔")
else:
    print(miss[miss > 0])

section("8. 幸福分數最高的前 10 國")
print(df.nlargest(10, "Happiness.Score")[["Happiness.Rank", "Country", "Happiness.Score"]]
      .to_string(index=False))

section("9. 幸福分數最低的後 10 國")
print(df.nsmallest(10, "Happiness.Score")[["Happiness.Rank", "Country", "Happiness.Score"]]
      .to_string(index=False))

# ---- 想印出「全部 155 國」時，把下面這行的註解拿掉 --------------------
# section("10. 全部國家一覽")
# print(df[["Happiness.Rank", "Country", "Happiness.Score"]].to_string(index=False))

section("完成")
print(f"資料來源：{CSV_PATH}")
