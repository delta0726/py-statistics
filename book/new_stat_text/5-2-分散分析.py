# **************************************************************************************
# Title   : あたらしいPythonで学ぶ統計学の教科書
# Chapter : 5 正規線形モデル
# Theme   : 2 分散分析
# Date    : 2022/05/04
# Page    : P302 - P318
# URL     : https://logics-of-blue.com/python-stats-book-support/
# **************************************************************************************


# ＜概要＞
# - 正規線形モデルにおける検定手法として用いられる分散分析(ANOVA)を学ぶ
#   --- エクセルの回帰分析のサマリーにも掲載されている
#   --- ANalysis Of VAriance


# ＜分散分析の前提条件＞
# - 母集団が正規分布に従うデータに対してのみ適用可能
# - 水準間で分散の値が異ならない


# ＜目次＞
# 0 分散分析とは
# 1 準備
# 2 データ作成
# 3 群間/郡内平方和の計算
# 4 群間/郡内分散とF値の計算
# 5 p値の計算
# 6 {statsmodels}による分散分析
# 7 モデル係数の解釈
# 8 回帰モデルによる分散分析


# 0 分散分析とは -----------------------------------------------------------------------

# ＜分散分析＞
# - 分散分析とは正規線形モデルにおいて幅広く用いられる検定手法
# - 分散分析は平均値の差を検定する手法のひとつ
#   --- 3つ以上のカテゴリの平均値に差があるかどうかを検定したい場合に使用する


# ＜検定の多重性＞
# - 検定を繰り返すことによって有意な結果が得られやすくなってしまう問題を検定の多重性という


# ＜分散分析の考え方＞
# - カテゴリごとの分布や平均値の水準に差が有意なあるかどうかを調べることを分散分析という
#   --- まずデータ変動を｢効果｣と｢誤差｣に分離する
#   --- F値(効果の分散/誤差の分散)を計算する


# ＜解釈＞
# - ｢効果｣とは天気がもたらす売上(Y)の変動のことを指す
# - ｢誤差｣とは天気(X)という変数を用いて説明することが出来なかった売上の変動
# - F値が大きければ誤差に比べて効果の影響が大きいと判断する




# 1 準備 -------------------------------------------------------------------------------

# ライブラリ
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm

from scipy import stats
from matplotlib import pyplot as plt


# データロード
beer = pd.read_csv("csv/5-1-1-beer.csv")


# 2 データ作成 -------------------------------------------------------------------------

# ＜ポイント＞
# - 結果を見やすくするため最小限のサンプル数でデータセットを作成する
#   --- cloudy/rainy/sunnyを2サンプルずつ


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


# 3 群間/郡内平方和の計算 ---------------------------------------------------------------

# ＜ポイント＞
# - 分散を計算するための準備として分散量(平方和)を計算しておく
#   --- 全変動を群間変動と郡内変動に分離する


# 天気の持つ影響
# --- データを郡内平均値に置換
effect = [7, 7, 3, 3, 11, 11]

# 群間の平方和
# --- 群間平均と全体平均の残差平方和
mu_effect = np.mean(effect)
squares_model = np.sum((effect - mu_effect) ** 2)
squares_model

# 群内の平方和
# --- 元データ - 郡内平均値
resid = weather_beer.beer - effect
squares_resid = np.sum(resid ** 2)
squares_resid


# 4 群間/郡内分散とF値の計算 ---------------------------------------------------------

# ＜ポイント＞
# - 群間/群内の分散を計算する際の母数には自由度を用いる
#   --- 群間 ：3 - 1 = 2  (曇/雨/晴)
#   --- 郡内 ：6 - 3 = 3  (サンプルサイズ=6, 水準数=3)


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

# F値
# --- 群間分散と群内分散の比率
f_ratio = variance_model / variance_resid
f_ratio


# 5 p値の計算 ---------------------------------------------------------------------

# ＜ポイント＞
# - p値はF分布の累積分布関数から計算することができる
#   --- 分散分析の目的はp値によりカテゴリ間に差が出ているかを判定すること


# p値
# --- p=0.025となって0.05以下となった
# --- 天気によって売上は有意に変化すると判断できる
1 - sp.stats.f.cdf(x=f_ratio, dfn=df_model, dfd=df_resid)


# 6 {statsmodels}による分散分析 --------------------------------------------------

# ＜ポイント＞
# - 正規線形モデルではカテゴリカル変数はダミー変数に変換して扱う
#   --- smf.ols()ではモデル内で自動でダミー変換されている
# - 分散分析は線形回帰モデルから計算する


# モデル構築
# --- 正規線形モデル
anova_model = smf.ols("beer ~ weather", data=weather_beer).fit()

# サマリー
# --- カテゴリカル変数が自動でダミー変換されていることを確認
anova_model.summary()

# 分散分析
# --- 一元配置の近似線形モデルのAnovaテーブル
sm.stats.anova_lm(anova_model, typ=2)


# 7 モデル係数の解釈 -------------------------------------------------------------

# ＜ポイント＞
# - モデルの当てはめ予測値は各水準の平均値と一致している


# 回帰係数
# --- 切片が7なので平均的に7本売れる
# --- 雨の日は-3本、晴れの日は+3本売れる
anova_model.params

# 予測値
# --- 訓練データに対する当てはめ結果
# --- モデルの当てはめ予測値は各水準の平均値と一致している
fitted = anova_model.fittedvalues

# 残差
anova_model.resid


# 8 回帰モデルによる分散分析 --------------------------------------------------------

# ＜ポイント＞
# - 線形回帰分析のサマリーには分散分析が含まれることが多い
#   --- F値やp値は回帰分析が適切に行われているかを評価する指標として扱われる


# データ確認
beer

# モデル推定
lm_model = smf.ols(formula="beer ~ temperature", data=beer).fit()

# サマリー
lm_model.summary()

# パラメータ設定
# --- モデルの自由度（切片と傾きで2、2-1）
# --- 残差の自由度（サンプルサイズ=30、30-2）
df_lm_model = 1
df_lm_resid = 28

# 効果の分散
lm_effect = lm_model.fittedvalues
mu = np.mean(lm_effect)
squares_lm_model = np.sum((lm_effect - mu) ** 2)
variance_lm_model = squares_lm_model / df_lm_model

# 誤差の分散
lm_resid = lm_model.resid
squares_lm_resid = np.sum(lm_resid ** 2)
variance_lm_resid = squares_lm_resid / df_lm_resid

# F比
f_value_lm = variance_lm_model / variance_lm_resid
f_value_lm

# ANOVA
# --- p値はほぼゼロ
sm.stats.anova_lm(lm_model, typ=2)

# 回帰分析サマリー
# --- OLS Regression ResultsにF値やp値が含まれている
lm_model.summary()
