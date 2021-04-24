# ***************************************************************************************
# Title     : あたらしいPythonで学ぶ統計学の教科書
# Chapter   : 5-2 分散分析
# Created by: Owner
# Created on: 2021/4/24
# Page      : P302 - P318
# ***************************************************************************************


# ＜概要＞


# ＜目次＞
# 0 準備
# 1 データ作成
# 2 群間/郡内平方和の計算
# 3 群間/郡内分散の計算
# 4 p値の計算
# 5 {statsmodels}による分散分析
# 6 モデル係数の解釈
# 7 回帰モデルによる分散分析


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import pandas as pd
import scipy as sp
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm

from scipy import stats
from matplotlib import pyplot as plt


# 1 データ作成 -------------------------------------------------------------------------

# リスト作成
weather = ["cloudy", "cloudy", "rainy", "rainy", "sunny", "sunny"]
beer = [6, 8, 2, 4, 10, 12]

# データフレーム作成
weather_beer = pd.DataFrame({"beer": beer, "weather": weather})
weather_beer

# プロット確認
# --- ボックスプロット
sns.boxplot(x="weather", y="beer", data=weather_beer, color="gray")
plt.show()

# データ集計
# --- グループごとの平均値
weather_beer.groupby("weather").mean()


# 2 群間/郡内平方和の計算 ---------------------------------------------------------------

# ＜ポイント＞
# - 最初はライブラリを使わずに一元配置分散分析を実装する

# 天気の持つ影響
effect = [7, 7, 3, 3, 11, 11]

# 群間の平方和
mu_effect = sp.mean(effect)
squares_model = sp.sum((effect - mu_effect) ** 2)
squares_model

# 残差
resid = weather_beer.beer - effect
resid

# 群内の平方和
squares_resid = sp.sum(resid ** 2)
squares_resid


# 3 群間/郡内分散の計算 --------------------------------------------------------------

# パラメータ設定
# --- 群間変動の自由度
# --- 群内変動の自由度
df_model = 2
df_resid = 3

# 群間の平均平方(分散)
variance_model = squares_model / df_model
variance_model

# 群内の平均平方(分散)
variance_resid = squares_resid / df_resid
variance_resid


# 4 p値の計算 ---------------------------------------------------------------------

# F値
# --- 群間分散と群内分散の比率
f_ratio = variance_model / variance_resid
f_ratio

# p値
# --- F分布の累積分布関数から計算
1 - sp.stats.f.cdf(x=f_ratio, dfn=df_model, dfd=df_resid)


# 5 {statsmodels}による分散分析 --------------------------------------------------

# モデル構築
# --- 正規線形モデル
anova_model = smf.ols("beer ~ weather", data=weather_beer).fit()

# 分散分析
# --- 一元配置の近似線形モデルのAnovaテーブル
sm.stats.anova_lm(anova_model, typ=2)


# 6 モデル係数の解釈 -------------------------------------------------------------

# 推定モデルの係数表示
anova_model.params

# 推定値
fitted = anova_model.fittedvalues

# 残差
anova_model.resid


# 7 回帰モデルによる分散分析 --------------------------------------------------------

# データ準備
beer = pd.read_csv("book/new_stat_text/csv/5-1-1-beer.csv")
beer

# モデル推定
lm_model = smf.ols(formula="beer ~ temperature", data=beer).fit()
lm_model

# パラメータ設定
# --- モデルの自由度
# --- 残差の自由度
df_lm_model = 1
df_lm_resid = 28

# 推定値
lm_effect = lm_model.fittedvalues

# 残差
lm_resid = lm_model.resid

# 気温の持つ効果の大きさ
mu = sp.mean(lm_effect)
squares_lm_model = sp.sum((lm_effect - mu) ** 2)
variance_lm_model = squares_lm_model / df_lm_model

# 残差の大きさ
squares_lm_resid = sp.sum(lm_resid ** 2)
variance_lm_resid = squares_lm_resid / df_lm_resid

# F比
f_value_lm = variance_lm_model / variance_lm_resid
f_value_lm

sm.stats.anova_lm(lm_model, typ=2)

lm_model.summary()
