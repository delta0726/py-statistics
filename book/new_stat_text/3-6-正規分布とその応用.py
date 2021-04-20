# ***************************************************************************************
# Title     : あたらしいPythonで学ぶ統計学の教科書
# Chapter   : 3-6 正規分布とその応用
# Created by: Owner
# Created on: 2021/4/20
# Page      : P178 - P189
# ***************************************************************************************


# ＜概要＞
# - 正規分布であることを活かせば、シミュレーションすることなく統計的性質の議論ができる
# - 確率密度関数の扱いとt分布について確認する


# ＜目次＞
# 0 準備
# 1 確率密度の実装
# 2 標本がある値以下となる割合
# 3 累積分布関数
# 4 パーセント点
# 5 t値の標準正規分布
# 6 t分布


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import numpy as np
import scipy as sp
import seaborn as sns

from scipy import stats
from matplotlib import pyplot as plt

# 描画セット
sns.set()


# 1 確率密度の実装 ----------------------------------------------------------------------

# 円周率
sp.pi

# 自然対数の底
sp.exp(1)

# パラメータ設定
# --- 数値：3
# --- 母平均：4
# --- 母標準偏差：0.8
x = 3
mu = 4
sigma = 0.8

# 確率密度
# --- 公式に基づいて定義
1 / (sp.sqrt(2 * sp.pi * sigma**2)) * sp.exp(-((x - mu)**2) / (2 * sigma**2))

# 確率密度
# --- 関数利用
stats.norm.pdf(x=x, loc=mu, scale=sigma)

# 確率密度
# --- 正規分布のインスタンス作成
# --- 値に応じた確率密度の算出
norm_dist = stats.norm(loc=mu, scale=sigma)
norm_dist.pdf(x=3)

# プロット作成
# --- 正規分布の確率密度
x_plot = np.arange(start=1, stop=7.1, step=0.1)
plt.plot(x_plot, stats.norm.pdf(x=x_plot, loc=mu, scale=sigma))
plt.show()


# 2 標本がある値以下となる割合 ---------------------------------------------------------

# ＜ポイント＞
# - 累積分布関数の元になる考え方を確認する

# パラメータ設定
# --- 数値：3
# --- 母平均：4
# --- 母標準偏差：0.8
x = 3
mu = 4
sigma = 0.8

# 正規乱数の生成
np.random.seed(1)
simulated_sample = stats.norm.rvs(loc=mu, scale=sigma, size=10000)
simulated_sample

# 割合の算出
# --- 個数ベースの割合
sp.sum(simulated_sample <= 3) / len(simulated_sample)


# 3 累積分布関数 -------------------------------------------------------------------

# ＜ポイント＞
# - 累積分布関数を一般化して考える
#   --- ccf(Cumulative Distribution Function)

# パラメータ設定
# --- 数値：3
# --- 母平均：4
# --- 母標準偏差：0.8
x = 3
mu = 4
sigma = 0.8

# 確率密度
# --- 上記の正規乱数の個数ベース割合を一般化したもの
stats.norm.cdf(loc=mu, scale=sigma, x=x)

# 平均の確率密度
# --- 正規分布は左右対称なので50％となる
stats.norm.cdf(loc=mu, scale=sigma, x=mu)


# 4 パーセント点 -------------------------------------------------------------------

# ＜ポイント＞
# - データがある値以下になる確率を｢下側確率｣という
#   --- 逆にある確率になる基準点を｢パーセント点｣という

# パラメータ設定
# --- 数値：3
# --- 母平均：4
# --- 母標準偏差：0.8
x = 3
mu = 4
sigma = 0.8

# パーセント点の算出
stats.norm.ppf(loc=mu, scale=sigma, q=0.025)

# 下側確率とパーセント点の関係
# --- 下側確率の算出
# --- 下側確率からパーセント点を算出
sitagawa = stats.norm.cdf(loc=mu, scale=sigma, x=x)
stats.norm.ppf(loc=mu, scale=sigma, q=sitagawa)

# 下側確率が50％となる点
# ---muに一致する
stats.norm.ppf(loc=mu, scale=sigma, q=0.5)


# 5 t値の標準正規分布 --------------------------------------------------------------

# パラメータ設定
# --- 母平均：4
# --- 母標準偏差：0.8
mu = 4
sigma = 0.8

# 乱数シード
np.random.seed(1)

# 配列作成
# --- t値を格納する入れ物
t_value_array = np.zeros(10000)

# インスタンス作成
# --- 正規分布クラス
norm_dist = stats.norm(loc=mu, scale=sigma)

# シミュレーション実行
# --- 10個の正規乱数を生成
# --- 標準誤差の算出(1サンプルあたりの分散量)
# --- t値の算出
for i in range(0, 10000):
    sample = norm_dist.rvs(size=10)
    sample_mean = sp.mean(sample)
    sample_std = sp.std(sample, ddof=1)
    sample_se = sample_std / sp.sqrt(len(sample))
    t_value_array[i] = (sample_mean - mu) / sample_se

# ヒストグラム作成
# --- t値
sns.distplot(t_value_array, color='black')

# 点線の重ね書き
x = np.arange(start=-8, stop=8.1, step=0.1)
y = stats.norm.pdf(x=x)
plt.plot(x, y, color='black', linestyle='dotted')
plt.show()


# 6 t分布 -----------------------------------------------------------------------

# ＜ポイント＞
# - 母集団が正規分布であるときのt値の標本分布を｢t分布｣という

# パラメータ設定
n = 10

# t分布のプロット
# --- 標準正規分布も併せてプロット
# --- t分布のほうが僅かに裾が広い（平均からの乖離が大きいサンプルが出やすい）
x = np.arange(start=-8, stop=8.1, step=0.1)
y = stats.norm.pdf(x=x)
t = stats.t.pdf(x=x, df=n-1)
plt.plot(x, y, color='black', linestyle='dotted')
plt.plot(x, t, color='black')
plt.show()

# シミュレーション結果と理論分布の比較
# --- 5の結果を使用
t = stats.t.pdf(x=x, df=n-1)
sns.distplot(t_value_array, color='black', norm_hist=True)
plt.plot(x, t, color='black', linestyle='dotted')
plt.show()
