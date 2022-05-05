# **************************************************************************************
# Title   : あたらしいPythonで学ぶ統計学の教科書
# Chapter : 5 正規線形モデル
# Theme   : 3 複数の説明変数を持つモデル
# Date    : 2022/05/04
# Page    : P319 - P337
# URL     : https://logics-of-blue.com/python-stats-book-support/
# **************************************************************************************


# ＜概要＞
# - 現実の世界では多数の要因を説明変数としてもつ重回帰分析を行う


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

# ＜ポイント＞
# - 本来複数の説明変数が必要なモデルで、あえて悪い分析例1変数のモデルを構築する
#   --- 店はpriceをweatherによってコントロールしているが、priceだけを取り出して分析している

# ＜解釈＞
# - Salesとは数量を意味する
# - 以下では｢価格を上げると売上(数量)が増える｣という結果が導かれる（直感と矛盾している）
#   --- 天気という交絡変数が存在するのを見落としている


# データ確認
sales.loc[:, ["sales", "price"]].head()

# モデル構築
# --- 単回帰モデル
lm_dame = smf.ols("sales ~ price", data=sales).fit()

# パラメータの確認
# --- priceの係数は0.33とプラスになっている
# --- 価格が高いほど売上が伸びる
lm_dame.params

# 分散分析
# --- p値は0.028と0.05を下回っている
# --- 価格の差は売上に影響を与えている
sm.stats.anova_lm(lm_dame, type=2)

# プロット作成
# --- 散布図 + 回帰直線
sns.lmplot(x="price", y="sales", data=sales,
           scatter_kws={"color": "black"},
           line_kws={"color": "black"})
plt.show()


# 3 説明変数同士の関係を調べる ---------------------------------------------------------

# ＜ポイント＞
# - この店は雨の日は売上が下がるので値引きして売っていた
#   --- 天気が同じだった日の商品価格が売上にもたらす影響を知りたい

# ＜多重検定に注意＞
# - 天気データでデータセットを分割して、それぞれに対して回帰分析を行うと多重検定問題が発生する
#   --- 各要因の影響を正しく判断するためには複数の説明変数を持つモデルを一発で推定しないといけない


# データ確認
# --- 2で見落としたweatherとpriceの関係を確認する
# --- 天気によって値段が異なることが分かる（ただし、一律の値段にはなっていない）
sales.loc[:, ["price", "weather"]]
pd.crosstab(index=sales['price'], columns=sales['weather'])

# データ集計
# --- グループ平均
# --- 晴れの日は売上が上がり、雨の日は売上が下がる
sales.groupby("weather").mean()

# プロット作成
# --- 晴れの日のほうが売上が伸びている
# --- 価格が高くなると売上は下がる
sns.lmplot(x="price", y="sales", data=sales,
           hue="weather", palette="gray")

plt.show()


# 4 複数の説明変数を持つモデル -------------------------------------------------------

# ＜ポイント＞
# - 複数の説明変数を用いたモデルを構築する
#   --- 価格が下がると売上(数量)が増加するという適切な結果が導き出された


# モデル構築
lm_sales = smf.ols("sales ~ weather + humidity + temperature + price", data=sales).fit()

# パラメータ
# --- priceの係数がマイナス
# --- 価格が上がると売上(数量)が下がることを示唆
lm_sales.params


# 5 悪い分析例：通常の分散分析で検定する ---------------------------------------------

# ＜ポイント＞
# - 素朴な分散分析(typ=1)は説明変数の順番を変えると結果が変わってしまう
#   --- 回帰分析の結果自体は同じ
#   --- Type2 ANOVA(typ=2)を使うと問題は発生しない


# 分散分析1
# --- 上記のモデルをそのまま分散分析
lm_sales = smf.ols("sales ~ weather + humidity + temperature + price", data=sales).fit()
sm.stats.anova_lm(lm_sales, typ=1).round(3)

# 分散分析2
# --- 上記のモデルの変数の順番を入れ替えて分散分析
# --- 分散分析1と結果が異なる点に注意
lm_sales_2 = smf.ols("sales ~ weather + temperature + humidity + price", data=sales).fit()
sm.stats.anova_lm(lm_sales_2, typ=1).round(3)

# 参考
# --- 回帰係数は同じ
lm_sales.params
lm_sales_2.params


# 6 回帰分析のt検定 ------------------------------------------------------------

# ＜ポイント＞
# - 回帰係数のt検定であれば順番入れ替えに伴うエラーは発生しない
#   --- 今回はカテゴリ変数が晴/雨の2つのみなので問題ない
#   --- 仮に曇があってカテゴリが3つなら検定の多重性問題が発生する


# モデル1
print(lm_sales.summary().tables[1])

# モデル2
print(lm_sales_2.summary().tables[1])


# 7 モデル選択と分散分析 ------------------------------------------------------

# ＜ポイント＞
# - Type1 ANOVAは複数の説明変数を持つ場合に1つずつ説明変数を増やしていく
#   --- 説明変数が増えることによって減少する残差平方和に着目している
#   --- 説明変数の順番に依存してしまう（以下の例で確認）


# Nullモデル *******************************************

# 残差平方和
# --- Nullモデル
mod_null = smf.ols("sales ~ 1", data=sales).fit()
resid_sq_null = np.sum(mod_null.resid ** 2)
resid_sq_null


# 天気モデル *********************************************

# 残差平方和
# --- 天気モデル（モデル1）
mod_1 = smf.ols("sales ~ weather", data=sales).fit()
resid_sq_1 = np.sum(mod_1.resid ** 2)
resid_sq_1

# 残差平方和の差分
resid_sq_null - resid_sq_1

# 分散分析表
sm.stats.anova_lm(mod_1).round(3)


# 天気+湿度モデル *****************************************

# 残差平方和
# --- 天気+湿度モデル（モデル2）
mod_2 = smf.ols("sales ~ weather + humidity", data=sales).fit()
resid_sq_2 = np.sum(mod_2.resid ** 2)
resid_sq_2

# 残差平方和の差分
resid_sq_1 - resid_sq_2

# 分散分析表
sm.stats.anova_lm(mod_2).round(3)


# 天気+気温モデル *****************************************

# 残差平方和
# --- 天気+湿度モデル（モデル2-2）
mod_2_2 = smf.ols("sales ~ weather + temperature", data=sales).fit()
resid_sq_2_2 = np.sum(mod_2_2.resid ** 2)
resid_sq_2_2


# 天気+気温+湿度モデル *****************************************

# 残差平方和
# --- 天気+気温+湿度モデル（モデル3-2）
mod_3_2 = smf.ols("sales ~ weather + temperature + humidity", data=sales).fit()
resid_sq_3_2 = np.sum(mod_3_2.resid ** 2)
resid_sq_3_2


# 残差平方和の差分
resid_sq_2_2 - resid_sq_3_2

# 分散分析表
sm.stats.anova_lm(mod_3_2).round(3)


# 8 Type2 ANOVA -------------------------------------------------------------------

# ＜ポイント＞
# - 特定の説明変数を除いた際の分散量の差分とType2 ANOVAの除外した分散量は一致する
#   --- Type2 ANOVAのほうが直感的で解釈しやすい


# 残差平方和
# --- 全ての変数を入れたモデル
mod_full = smf.ols("sales ~ weather + temperature + humidity + price", data=sales).fit()
resid_sq_full = np.sum(mod_full.resid ** 2)
resid_sq_full

# 残差平方和
# --- 湿度だけ除いたモデル
mod_non_humi = smf.ols("sales ~ weather + temperature + price", data=sales).fit()
resid_sq_non_humi = np.sum(mod_non_humi.resid ** 2)
resid_sq_non_humi

# 残差平方和の差分
resid_sq_non_humi - resid_sq_full

# 分散分析
sm.stats.anova_lm(mod_full,typ=2).round(3)

# 2つのモデルを直接比較
mod_full.compare_f_test(mod_non_humi)


# 9 変数選択とモデル解釈 --------------------------------------------------------------

# ＜ポイント＞
# - 分散分析のp値が全て0.05以下となるモデルを探す
#   --- 湿度を抜いたモデルが該当する


# 分散分析
# --- p値が全て0.05以下となるモデルを探す（適切なモデル）
sm.stats.anova_lm(mod_non_humi, typ=2).round(3)

# 回帰係数
# --- 適切なモデルを導いてから回帰係数を解釈する
mod_non_humi.params


# 10 AICによる変数選択 --------------------------------------------------------------

# ＜ポイント＞
# - AICはモデルごとに計算されて、AICが低いモデルのほうが良いモデルと判断される
#   --- AICは複数のカテゴリを持つ変数であっても検定の多重性を気にしなくてよい（使い勝手が良い）
#   --- 湿度は除外したほうが良いという結果

# AICの計算
# --- 全変数モデル
# --- 湿度抜きモデル
mod_full.aic
mod_non_humi.aic
