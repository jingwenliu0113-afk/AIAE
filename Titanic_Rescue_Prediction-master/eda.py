# -*- coding: utf-8 -*-
"""
Titanic 資料集 EDA（探索性資料分析）腳本
========================================
直接在 PyCharm 按右鍵 Run 即可執行。
會在終端印出資料摘要，並把圖表存到 ./eda_output/ 資料夾。

資料欄位說明：
    PassengerId : 乘客編號（僅識別用）
    Survived    : 是否生還  0=罹難, 1=生還（← 預測目標，test.csv 沒有這欄）
    Pclass      : 船艙等級  1=頭等, 2=二等, 3=三等（社經地位代理變數）
    Name        : 姓名（內含 Mr/Mrs/Miss 等頭銜，可抽取特徵）
    Sex         : 性別
    Age         : 年齡（有缺失值）
    SibSp       : 同行的兄弟姊妹／配偶人數
    Parch       : 同行的父母／子女人數
    Ticket      : 船票號碼
    Fare        : 票價
    Cabin       : 船艙號碼（大量缺失）
    Embarked    : 登船港口  C=Cherbourg, Q=Queenstown, S=Southampton
"""

import os
import sys

# 讓終端輸出使用 UTF-8，避免 Windows 主控台顯示中文變成亂碼
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

# ---- 基本設定 ----------------------------------------------------------
pd.set_option("display.max_columns", None)   # 印出時不省略欄位
pd.set_option("display.width", 200)

# 先設定主題，再套字型（sns.set_theme 會覆寫 rcParams，所以字型要放後面）
sns.set_theme(style="whitegrid")

# 中文字型：從系統實際安裝的字型中挑一個可用的，避免圖表中文變成方框。
# 直接指定字型名而不驗證是沒用的 —— matplotlib 找不到時會靜默 fallback 回 Arial。
_installed = {f.name for f in font_manager.fontManager.ttflist}
for _font in ["Microsoft JhengHei", "Microsoft YaHei", "MingLiU", "SimHei", "SimSun"]:
    if _font in _installed:
        matplotlib.rcParams["font.sans-serif"] = [_font]
        matplotlib.rcParams["font.family"] = "sans-serif"
        print(f"[字型] 使用中文字型：{_font}")
        break
else:
    print("[字型] 警告：找不到可用的中文字型，圖表中文可能顯示為方框。")
matplotlib.rcParams["axes.unicode_minus"] = False   # 負號正常顯示

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "eda_output")
os.makedirs(OUT_DIR, exist_ok=True)


def section(title: str) -> None:
    """在終端印出分隔標題。"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_fig(name: str) -> None:
    """存檔並關閉目前的圖，避免視窗一直彈出。"""
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  圖已存檔 -> {path}")


# ---- 讀取資料 ----------------------------------------------------------
train = pd.read_csv(os.path.join(HERE, "train.csv"))
test = pd.read_csv(os.path.join(HERE, "test.csv"))

# =======================================================================
# 1. 資料整體樣貌
# =======================================================================
section("1. 資料形狀 (rows, columns)")
print(f"train : {train.shape}")
print(f"test  : {test.shape}")

section("2. 前 5 筆 train 資料")
print(train.head())

section("3. 欄位型別與非空數量 (info)")
train.info()

section("4. 數值欄位統計摘要 (describe)")
print(train.describe())

section("5. 類別欄位統計摘要")
# pandas 3.0 起字串欄位為 'str' dtype，明確同時納入 object 與 str 以相容新舊版本
print(train.describe(include=["object", "str"]))

# =======================================================================
# 2. 缺失值分析
# =======================================================================
section("6. 缺失值統計（train）")
miss = train.isnull().sum()
miss = miss[miss > 0].sort_values(ascending=False)
miss_pct = (miss / len(train) * 100).round(2)
print(pd.DataFrame({"缺失數量": miss, "缺失比例(%)": miss_pct}))

section("7. 缺失值統計（test）")
miss_t = test.isnull().sum()
miss_t = miss_t[miss_t > 0].sort_values(ascending=False)
print(pd.DataFrame({"缺失數量": miss_t,
                    "缺失比例(%)": (miss_t / len(test) * 100).round(2)}))

# 缺失值熱力圖
plt.figure(figsize=(10, 5))
sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("train.csv 缺失值分佈（黃色=缺失）")
save_fig("01_missing_heatmap.png")

# =======================================================================
# 3. 目標變數：生存率
# =======================================================================
section("8. 整體生存率")
surv_rate = train["Survived"].mean()
print(train["Survived"].value_counts())
print(f"整體生還比例：{surv_rate:.2%}（0=罹難 1=生還）")

plt.figure(figsize=(5, 4))
sns.countplot(data=train, x="Survived", palette="Set2", hue="Survived", legend=False)
plt.title("生存人數分佈 (0=罹難, 1=生還)")
save_fig("02_survived_count.png")

# =======================================================================
# 4. 各特徵 vs 生存率
# =======================================================================
def survival_by(col: str) -> None:
    """印出並繪製某欄位分組後的生存率。"""
    print(f"\n--- {col} 分組生存率 ---")
    rate = train.groupby(col)["Survived"].agg(["mean", "count"])
    rate.columns = ["生存率", "人數"]
    print(rate)


section("9. 性別 / 艙等 / 登船港 對生存率的影響")
for c in ["Sex", "Pclass", "Embarked"]:
    survival_by(c)

# 三合一長條圖
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["Sex", "Pclass", "Embarked"]):
    sns.barplot(data=train, x=col, y="Survived", ax=ax,
                palette="Set2", hue=col, legend=False, errorbar=None)
    ax.set_title(f"{col} vs 生存率")
    ax.set_ylabel("生存率")
save_fig("03_survival_by_category.png")

# =======================================================================
# 5. 年齡與票價分佈
# =======================================================================
section("10. 年齡 / 票價 依生存狀態的分佈")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(data=train, x="Age", hue="Survived", bins=30,
             kde=True, palette="Set1")
plt.title("年齡分佈 (依生存狀態)")

plt.subplot(1, 2, 2)
sns.histplot(data=train, x="Fare", hue="Survived", bins=40,
             kde=False, palette="Set1")
plt.title("票價分佈 (依生存狀態)")
save_fig("04_age_fare_dist.png")

# =======================================================================
# 6. 特徵工程一瞥：頭銜 & 家庭大小
# =======================================================================
section("11. 從 Name 抽取頭銜 (Title) 的生存率")
train["Title"] = train["Name"].str.extract(r",\s*([^\.]+)\.")
print(train.groupby("Title")["Survived"].agg(["mean", "count"])
      .sort_values("count", ascending=False))

section("12. 家庭大小 (FamilySize = SibSp + Parch + 1) 的生存率")
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
print(train.groupby("FamilySize")["Survived"].agg(["mean", "count"]))

plt.figure(figsize=(7, 4))
sns.barplot(data=train, x="FamilySize", y="Survived",
            palette="Set2", hue="FamilySize", legend=False, errorbar=None)
plt.title("家庭大小 vs 生存率")
save_fig("05_familysize.png")

# =======================================================================
# 7. 數值欄位相關性
# =======================================================================
section("13. 數值欄位相關係數矩陣")
num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]
corr = train[num_cols].corr()
print(corr.round(2))

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("數值特徵相關性")
save_fig("06_correlation.png")

# =======================================================================
section("完成")
print(f"所有圖表已輸出到：{OUT_DIR}")
print("在 PyCharm 中可直接打開 eda_output 資料夾檢視圖片。")
