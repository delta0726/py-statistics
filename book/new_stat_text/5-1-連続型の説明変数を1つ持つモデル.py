# ***************************************************************************************
# Title     : あたらしいPythonで学ぶ統計学の教科書
# Chapter   : 5-1 連続型の説明変数を1つ持つモデル
# Created by: Owner
# Created on: 2021/4/23
# Page      : P280 - P301
# ***************************************************************************************


# ＜概要＞
# - {statsmodels}を用いて線形回帰モデルを構築する


# ＜目次＞
# 0 準備
# 1 線形回帰モデルの構築
# 2 AICによるモデル選択
# 3 回帰モデルの要素の操作
# 4 モデルの説明力
# 5 回帰モデルに関連するプロット


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

from scipy import stats
from matplotlib import pyplot as plt

import statsmodels.formula.api as smf
import statsmodels.api as sm

# 描画セット
sns.set()

# データ準備
beer = pd.read_csv("book/new_stat_text/csv/5-1-1-beer.csv")
beer.head()

# プロット確認
sns.jointplot(x="temperature", y="beer", data=beer, color="black")
plt.show()


# 1 線形回帰モデルの構築 -----------------------------------------------------------------

# モデル構築
# --- 正規線形モデル
lm_model = smf.ols(formula="beer ~ temperature", data=beer).fit()
lm_model

# サマリー
lm_model.summary()


# 2 AICによるモデル選択 ---------------------------------------------------------------

# Nullモデルの構築
# --- 切片項のみのモデル
# --- ベンチマークとして用いる
null_model = smf.ols(formula="beer ~ 1", data=beer).fit()

# AICの確認
# --- Nullモデル
# --- 線形回帰モデル
null_model.aic
lm_model.aic

# AICの計算
# --- 対数尤度
# --- 説明変数の数
# --- AICの公式に基づく計算
lm_model.llf
lm_model.df_model
-2 * (lm_model.llf - (lm_model.df_model + 1))


# 3 回帰モデルの要素の操作 --------------------------------------------------------------------

# 回帰直線の図示
sns.lmplot(x="temperature", y="beer", data=beer,
           scatter_kws={"color": "black"},
           line_kws={"color": "black"})
plt.show()

# パラメータ表示
lm_model.params

# モデルの予測
# --- インサンプルの推定値
lm_model.predict()

# モデルの予測
# --- アウトオブサンプルの予測値
lm_model.predict(pd.DataFrame({"temperature": [0]}))
lm_model.predict(pd.DataFrame({"temperature": [20]}))

# 手動計算
beta0 = lm_model.params.Intercept
beta1 = lm_model.params.temperature
temperature = 20
beta0 + beta1 * temperature

# 残差の取得
resid = lm_model.resid
resid

# 手動計算
beta0 = lm_model.params.Intercept
beta1 = lm_model.params.temperature
beer.temperature
y = beer.beer
y_hat = beta0 + beta1 * beer.temperature
y - y_hat


# 4 モデルの説明力 ------------------------------------------------------------------------

# 決定係数 *********************************************

# モデル結果から取得
lm_model.rsquared

# 公式に基づいて計算
y = beer.beer
mu = sp.mean(y)
yhat = lm_model.predict()
sp.sum((yhat - mu) ** 2) / sp.sum((y - mu) ** 2)

# 決定係数の分解
# --- データ全体の変動
sp.sum((yhat - mu) ** 2) + sum(resid ** 2)
sp.sum((y - mu) ** 2)

# 分解公式からの計算
1 - sp.sum(resid ** 2) / sp.sum((y - mu) ** 2)


# 自由度調整済み決定係数 ********************************

# モデル結果から取得
lm_model.rsquared_adj

# 公式に基づいて計算
n = len(beer.beer)
s = 1
1 - ((sp.sum(resid ** 2) / (n - s - 1)) / (sp.sum((y - mu) ** 2) / (n - 1)))


# 5 回帰モデルに関連するプロット -------------------------------------------------------------

# 残差のヒストグラム
# --- 残差はゼロを中心に正規分布を描く
sns.distplot(resid, color="black")
plt.show()

# 散布図
# --- X軸：予測値
# --- Y軸：残差
sns.jointplot(x=lm_model.fittedvalues, y=resid,
              joint_kws={"color": "black"},
              marginal_kws={"color": "black"})
plt.show()

# Q-Qプロット
fig = sm.qqplot(resid, line="s")
plt.show()

# Q-Qプロット(計算プロセス)
# --- 残差を昇順に並び替える
# --- 最も小さいデータの分位点(30サンプル)
resid_sort = resid.sort_values()
nobs = len(resid_sort)
cdf = np.arange(1, nobs + 1) / (nobs + 1)
ppf = stats.norm.ppf(cdf)
ppf
