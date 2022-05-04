# **************************************************************************************
# Title   : あたらしいPythonで学ぶ統計学の教科書
# Chapter : 5 正規線形モデル
# Theme   : 1 連続型の説明変数を1つ持つモデル
# Date    : 2022/05/04
# Page    : P280 - P301
# URL     : https://logics-of-blue.com/python-stats-book-support/
# **************************************************************************************


# ＜概要＞
# - {statsmodels}を用いて線形回帰モデルを構築する


# ＜目次＞
# 0 準備
# 1 線形回帰モデルの構築
# 2 AICによるモデル選択
# 3 回帰モデルの要素の操作
# 4 モデルによる予測
# 5 予測値と実測値の残差
# 6 決定係数
# 7 決定係数の分解
# 8 回帰モデルに関連するプロット
# 9 ダービン・ワトソン統計量


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from matplotlib import pyplot as plt

import statsmodels.formula.api as smf
import statsmodels.api as sm


# 描画セット
sns.set()

# データ準備
beer = pd.read_csv("csv/5-1-1-beer.csv")
beer.head()

# プロット確認
sns.jointplot(x="temperature", y="beer", data=beer, color="black")
plt.show()


# 1 線形回帰モデルの構築 -----------------------------------------------------------------

# ＜ポイント＞
# - 正規線形モデルを線形回帰モデルで定義する
# - 1変数の場合は散布図でも解釈できるが変数が多くなると回帰係数などを見るほうが解釈しやすくなる


# モデル構築
# --- 正規線形モデル
lm_model = smf.ols(formula="beer ~ temperature", data=beer).fit()
lm_model

# サマリー
lm_model.summary()


# 2 AICによるモデル選択 ---------------------------------------------------------------

# ＜ポイント＞
# - 説明変数が1つしかないためNullモデルとのAICで比較することになる
# - AICの水準自体には意味はなくモデル間の大小関係を比較するのに用いる
#   --- AICのパラメータ数(k)には局外パラメータを含むかどうかなど複数の計算定義がある
#   --- 同一ライブラリのAICを用いるのが無難


# Nullモデルの構築
# --- 切片項のみのモデル
# --- ベンチマークとして用いる
null_model = smf.ols(formula="beer ~ 1", data=beer).fit()

# AICの確認
# --- Nullモデル(227.94)
# --- 線形回帰モデル(208.91)
null_model.aic
lm_model.aic

# AICの計算
# --- 対数尤度
# --- 説明変数の数
# --- AICの公式に基づく計算（AIC = -2logL + 2k）
logL = lm_model.llf
k = lm_model.df_model
-2 * (logL - k)


# 3 回帰モデルの要素の操作 --------------------------------------------------------------------

# ＜ポイント＞
# - 回帰直線を図示するだけであればseabornを用いることで実現することができる


# 回帰直線の図示
sns.lmplot(x="temperature", y="beer", data=beer,
           scatter_kws={"color": "black"},
           line_kws={"color": "black"})
plt.show()

# パラメータ表示
lm_model.params


# 4 モデルによる予測 -------------------------------------------------------------------------

# ＜ポイント＞
# - モデルの目的のひとつである予測を行う


# モデルによる予測
# --- Xに何も指定しない場合はインサンプルの推定値が出力される
lm_model.predict()

# モデルによる予測
# --- Xの値を指定して予測値
lm_model.predict(pd.DataFrame({"temperature": [0]}))
lm_model.predict(pd.DataFrame({"temperature": [20]}))

# 手動計算
beta0 = lm_model.params.Intercept
beta1 = lm_model.params.temperature
temperature = 20
beta0 + beta1 * temperature


# 5 予測値と実測値の残差 ---------------------------------------------------------------

# ＜ポイント＞
# - 予測値と実測値の差分は残差として定義される
#   --- モデルの予測精度の評価などには残差を用いる
#   --- 正規線形モデルの場合は残差は平均0の正規分布に従う


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


# 6 決定係数 ------------------------------------------------------------------------

# ＜ポイント＞
# - 決定係数は予測値と実測値の平均値からの乖離から算出している


# 決定係数
# --- モデル結果から取得
lm_model.rsquared

# 決定係数
# --- 公式に基づいて計算
y = beer.beer
mu = np.mean(y)
yhat = lm_model.predict()
np.sum((yhat - mu) ** 2) / np.sum((y - mu) ** 2)


# 7 決定係数の分解 ----------------------------------------------------------------------

# ＜ポイント＞
# - 決定係数はモデルで説明できる全変動量と残差平方和に分解することができる
# - 自由度調整済み決定係数は説明変数が増えることに対して罰則を課したもの
#   --- 説明変数が増えると決定係数は増加するため


# 決定係数の分解
# --- データ全体の変動
ESS = np.sum((yhat - mu) ** 2)
RSS = sum(resid ** 2)
TSS = np.sum((y - mu) ** 2)

# 総変動量
ESS + RSS
TSS

# 決定係数の分解
# --- 分解公式からの計算
1 - RSS / TSS
1 - np.sum(resid ** 2) / np.sum((y - mu) ** 2)

# 自由度調整済み決定係数
# --- モデル結果から取得
lm_model.rsquared_adj

# 自由度調整済み決定係数
# --- 公式に基づいて計算
n = len(beer.beer)
s = 1
1 - (RSS / (n - s - 1)) / (TSS/ (n - 1))
1 - ((np.sum(resid ** 2) / (n - s - 1)) / (np.sum((y - mu) ** 2) / (n - 1)))


# 8 回帰モデルに関連するプロット -------------------------------------------------------------

# ＜ポイント＞
# - 回帰モデルを評価する際には残差を用いたプロットが使われる
#   --- 残差ヒストグラム（残差の全体像の把握）
#   --- 残差と予測値の散布図（残差の傾向確認）
#   --- Q-Qプロット（予測の正規性の確認）


# 残差ヒストグラム
# --- 残差はゼロを中心に正規分布を描く
sns.displot(resid, color="black")
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

# Q-Qプロットの考え方
# --- 残差の理論上の分位点と実測の分位点の乖離を比較する
# --- 残差を昇順に並び替える
# --- 最も小さいデータの分位点(30サンプル)
resid_sort = resid.sort_values()
nobs = len(resid_sort)
cdf = np.arange(1, nobs + 1) / (nobs + 1)
ppf = stats.norm.ppf(cdf)
ppf


# 9 ダービン・ワトソン統計量 ----------------------------------------------------------

# ＜ポイント＞
# - ダービン・ワトソン統計量は残差の自己相関をチェックする指標
#   --- 特に時系列データを扱う場合には2前後であることをチェックする
#   --- 残差に自己相関があると係数の信頼性がなくなる（見せかけの回帰）

# 残差の統計量の確認
lm_model.summary()