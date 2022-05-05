# ***********************************************************************************************
# Title   : あたらしいPythonで学ぶ統計学の教科書
# Chapter : 6 一般化線形モデル
# Theme   : 5 ポアソン回帰
# Date    : 2022/05/05
# Page    : P378 - P385
# URL     : https://logics-of-blue.com/python-stats-book-support/
# ***********************************************************************************************


# ＜概要＞
# - ポアソン回帰はロジスティック回帰の確率分布とリンク関数を切り替えるだけで計算することができる
#   --- 一般化線形モデルはパーツの切り替えのみで多用なパターンに対応することを確認


# ＜目次＞
# 0 準備
# 1 モデル構築
# 2 モデル選択
# 3 回帰曲線のプロット
# 4 回帰係数の解釈


# 0 準備 -------------------------------------------------------------------------------

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
beer = pd.read_csv("csv/6-5-1-poisson-regression.csv")

# データ確認
beer


# 1 モデル構築 -------------------------------------------------------------

# ＜ポイント＞
# - ポアソン回帰とは確率分布にポアソン分布を用いてリンク関数に対数関数を用いた一般化線形モデル
# - ビール販売個数をYとしてポアソン回帰モデルを構築する


# モデル化
mod_pois = smf \
    .glm(formula="beer_number ~ temperature", data=beer,
         family=sm.families.Poisson()) \
    .fit()

# サマリー
mod_pois.summary()


# 2 モデル選択 ---------------------------------------------------------------

# ＜ポイント＞
# - AICを用いてモデル選択を行う
#   --- Nullモデルと比較することでtemperatureを選択する有意性を確認する

# Nullモデル
mod_pois_null = smf\
    .glm(formula="beer_number ~ 1", data=beer,
         family=sm.families.Poisson())\
    .fit()

# AICの比較
# --- Nullモデル(223.36)
# --- 変数入りモデル(119.34)
mod_pois_null.aic
mod_pois.aic


# 3 回帰曲線のプロット ---------------------------------------------------

# ＜ポイント＞
# - ポアソン回帰はlmplotで作成することができないため重ね書きをする


# 予測値の作成
x_plot = np.arange(0, 37)

# 予測
pred = mod_pois.predict(pd.DataFrame({"temperature": x_plot}))

# プロット作成
# --- 回帰直線を入れないlmplot
sns.lmplot(y="beer_number", x="temperature",
           data=beer, fit_reg=False,
           scatter_kws={"color": "black"})
# 回帰曲線を上書き
plt.plot(x_plot, pred, color="black")

# プロット表示
plt.show()


# 4 回帰係数の解釈 ------------------------------------------------------

# ＜ポイント＞
# - リンク関数が恒等関数でない場合は回帰係数の解釈が変わる
#   --- ポアソン回帰はリンク関数に対数関数を使用しているので｢足し算｣が｢掛け算｣になっている


# 気温が1度の時の販売個数の期待値
exp_val_1 = pd.DataFrame({"temperature": [1]})
pred_1 = mod_pois.predict(exp_val_1)

# 気温が2度の時の販売個数の期待値
exp_val_2 = pd.DataFrame({"temperature": [2]})
pred_2 = mod_pois.predict(exp_val_2)

# 対数尤度比
# --- 気温が1度気温が1度上がると、販売個数は何倍になるか
pred_2 / pred_1

# 対数尤度比
# --- 回帰係数のexpをとる
np.exp(mod_pois.params["temperature"])
