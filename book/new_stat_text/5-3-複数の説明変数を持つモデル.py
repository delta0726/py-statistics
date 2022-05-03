# **************************************************************************************
# Title   : あたらしいPythonで学ぶ統計学の教科書
# Chapter : 5 正規線形モデル
# Theme   : 3 複数の説明変数を持つモデル
# Date    : 2022/05/04
# Page    : P319 - P337
# URL     : https://logics-of-blue.com/python-stats-book-support/
# **************************************************************************************


# ＜概要＞


# ＜目次＞
# 0 準備
# 1 データ可視化
# 2 悪い分析例：変数が1つだけのモデルを作る
# 3 説明変数同士の関係を調べる
# 4 複数の説明変数を持つモデル
# 5 悪い分析例：通常の分散分析で検定する
# 6 回帰分析のt検定
# 7 モデル選択と分散分析
# 8 Type2 ANOVA
# 9 変数選択とモデル解釈
# 10 AICによる変数選択


# 0 準備 -------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm

from matplotlib import pyplot as plt

# 描画セット
sns.set()

# データ準備
sales = pd.read_csv("csv/5-3-1-lm-model.csv")
sales


# 1 データ可視化 ----------------------------------------------------------------------

# 散布図行列
sns.pairplot(data=sales, hue="weather", palette="gray")
plt.show()


# 2 悪い分析例：変数が1つだけのモデルを作る ----------------------------------------------

# モデル構築
# --- 単回帰モデル
lm_dame = smf.ols("sales ~ price", data=sales).fit()

# パラメータの確認
lm_dame.params

# 分散分析
sm.stats.anova_lm(lm_dame, type=2)

# プロット作成
sns.lmplot(x="price", y="sales", data=sales,
           scatter_kws={"color": "black"},
           line_kws={"color": "black"})
plt.show()


# 3 説明変数同士の関係を調べる ---------------------------------------------------------

# データ集計
# --- グループ平均
# --- 雨の日は売上が下がる
sales.groupby("weather").mean()

# プロット作成
sns.lmplot(x="price", y="sales", data=sales,
           hue="weather", palette="gray")

plt.show()


# 4 複数の説明変数を持つモデル -------------------------------------------------------

# モデル構築
lm_sales = smf.ols("sales ~ weather + temperature + price", data=sales).fit()

# パラメータ
# --- priceの係数がマイナス
# --- 価格が上がると売上が下がることを示唆
lm_sales.params


# 5 悪い分析例：通常の分散分析で検定する ---------------------------------------------

# 分散分析
# --- Type1 ANOVA
sm.stats.anova_lm(lm_sales, typ=1).round(3)

# 説明変数の順番を変える
lm_sales_2 = smf.ols("sales ~ weather + temperature + humidity + price",
                     data=sales).fit()

# 結果
sm.stats.anova_lm(lm_sales_2, typ=1).round(3)


# 6 回帰分析のt検定 ------------------------------------------------------------

# モデル1
print(lm_sales.summary().tables[1])

# モデル2
print(lm_sales_2.summary().tables[1])


# 7 モデル選択と分散分析 ------------------------------------------------------

# 残差平方和
# --- Nullモデル
mod_null = smf.ols("sales ~ 1", data=sales).fit()
resid_sq_null = sp.sum(mod_null.resid ** 2)
resid_sq_null

# 天気モデル *********************************************

# 残差平方和
# --- 天気モデル（モデル1）
mod_1 = smf.ols("sales ~ weather", data=sales).fit()
resid_sq_1 = sp.sum(mod_1.resid ** 2)
resid_sq_1

# 残差平方和の差分
resid_sq_null - resid_sq_1

# 分散分析表
sm.stats.anova_lm(mod_1).round(3)


# 天気+湿度モデル *****************************************

# 残差平方和
# --- 天気+湿度モデル（モデル2）
mod_2 = smf.ols("sales ~ weather + humidity", data=sales).fit()
resid_sq_2 = sp.sum(mod_2.resid ** 2)
resid_sq_2

# 残差平方和の差分
resid_sq_1 - resid_sq_2

# 分散分析表
sm.stats.anova_lm(mod_2).round(3)


# 天気+気温モデル *****************************************

# 残差平方和
# --- 天気+湿度モデル（モデル2-2）
mod_2_2 = smf.ols("sales ~ weather + temperature", data=sales).fit()
resid_sq_2_2 = sp.sum(mod_2_2.resid ** 2)
resid_sq_2_2


# 天気+気温+湿度モデル *****************************************

# 残差平方和
# --- 天気+気温+湿度モデル（モデル3-2）
mod_3_2 = smf.ols("sales ~ weather + temperature + humidity", data=sales).fit()
resid_sq_3_2 = sp.sum(mod_3_2.resid ** 2)
resid_sq_3_2


# 残差平方和の差分
resid_sq_2_2 - resid_sq_3_2

# 分散分析表
sm.stats.anova_lm(mod_3_2).round(3)


# 8 Type2 ANOVA -------------------------------------------------------------------

# 残差平方和
# --- 全ての変数を入れたモデル
mod_full = smf.ols("sales ~ weather + temperature + humidity + price",
                   data=sales).fit()
resid_sq_full = sp.sum(mod_full.resid ** 2)
resid_sq_full

# 残差平方和
# --- 湿度だけ除いたモデル
mod_non_humi = smf.ols("sales ~ weather + temperature + price",
                       data=sales).fit()
resid_sq_non_humi = sp.sum(mod_non_humi.resid ** 2)
resid_sq_non_humi

# 残差平方和の差分
resid_sq_non_humi - resid_sq_full

# 分散分析
sm.stats.anova_lm(mod_full,typ=2).round(3)

# 2つのモデルを直接比較
mod_full.compare_f_test(mod_non_humi)


# 9 変数選択とモデル解釈 --------------------------------------------------------------

sm.stats.anova_lm(mod_non_humi, typ=2).round(3)

mod_non_humi.params


# 10 AICによる変数選択 --------------------------------------------------------------

# 全変数モデル
mod_full.aic

# 湿度抜きモデル
mod_non_humi.aic
