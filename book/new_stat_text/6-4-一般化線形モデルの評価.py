# ***********************************************************************************************
# Title   : あたらしいPythonで学ぶ統計学の教科書
# Chapter : 6 一般化線形モデル
# Theme   : 4 一般化線形モデルの評価
# Date    : 2022/05/00
# Page    : P370 - P377
# URL     : https://logics-of-blue.com/python-stats-book-support/
# ***********************************************************************************************


# ＜概要＞
# - 正規線形モデルと同様に一般化線形モデルでも残差のチェックは欠かせない
#   --- 母集団が正規分布でない場合は残差の扱い方が大きく異なる


# ＜目次＞
# 0 準備
# 1 モデル構築
# 2 ピアソン残差の計算
# 3 devianceの計算


# 0 準備 --------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

from matplotlib import pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm

# 描画セット
sns.set()

# データロード
test_result = pd.read_csv("csv/6-3-1-logistic-regression.csv")


# 1 モデル構築 ---------------------------------------------------------

# ＜ポイント＞
# - 目的変数がバイナリ変数のためロジスティク回帰を構築する


# データ確認
test_result

# クロス集計
pd.crosstab(test_result.loc[:, "hours"], test_result.loc[:, "result"])

# モデル構築
mod_glm = smf \
    .glm("result ~ hours", data=test_result, family=sm.families.Binomial()) \
    .fit()

# サマリー
mod_glm.summary()


# 2 ピアソン残差の計算 -----------------------------------------------------

# ＜ポイント＞
# - ピアソン残差は残差を分布の標準偏差で割ったもの
# - ピアソン残差の平方和はピアソンカイ二乗統計量となる


# 残差の要素
# --- 予測データ（クラス確率）
# --- 実測データ（1/0データ）
pred = mod_glm.predict()
y = test_result.result

# ピアソン残差
# --- 定義より計算（分母は二項分布の標準偏差）
# --- モデルより抽出
peason_resid = (y - pred) / sp.sqrt(pred * (1 - pred))
peason_resid
mod_glm.resid_pearson

# ピアソンカイ二乗値
# --- ピアソン残差の2乗和
# --- モデルより抽出
sp.sum(mod_glm.resid_pearson ** 2)
mod_glm.pearson_chi2


# 3 devianceの計算 -----------------------------------------------------------

# ＜ポイント＞
# - モデルの適合度を評価する指標にdevienceがある
#   --- devienceは残差平方和を尤度の考え方で表現したもの
#   --- devienceが大きいとモデルの当てはまりが悪いとみなされる


# 残差の要素
# --- 予測データ（クラス確率）
# --- 実測データ（1/0データ）
pred = mod_glm.predict()
y = test_result.result

# devience
# --- 合否を完全に予測できた時の対数尤度との差異
# --- deviance残差
resid_tmp = 0 - np.log(sp.stats.binom.pmf(k=y, n=1, p=pred))
deviance_resid = np.sqrt(2 * resid_tmp) * np.sign(y - pred)
deviance_resid

# devience
# --- 定義より計算（分母は二項分布の標準偏差）
# --- モデルより抽出
mod_glm.resid_deviance

# devianceの平方和
np.sum(mod_glm.resid_deviance ** 2)
