# ***********************************************************************************************
# Title   : あたらしいPythonで学ぶ統計学の教科書
# Chapter : 6 一般化線形モデル
# Theme   : 1 様々な確率分布
# Date    : 2022/05/05
# Page    : P340 - P350
# URL     : https://logics-of-blue.com/python-stats-book-support/
# ***********************************************************************************************


# ＜概要＞
# - 一般化線形モデルは正規分布以外の確率分布を扱うことができる
#   --- それぞれの確率分布の特徴を理解しておく必要がある


# ＜準備＞
# 0 準備
# 1 二項分布
# 2 ポアソン分布
# 3 二項分布とポアソン分布の関係


# 0 準備 ---------------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

from matplotlib import pyplot as plt
import seaborn as sns


# プロット設定
sns.set()


# 1 二項分布 -----------------------------------------------------------------------

# ＜ポイント＞
# - 裏/表など2値しかとらないベルヌーイ試行に基づく確率分布を二項分布という
#   --- N回の独立した試行のうち成功確率がpで成功した回数(m)が従う離散型の確率分布


# 確率質量関数
# --- 成功確率が0.5の場合に2回の試行で1回成功する確率
sp.stats.binom.pmf(k = 1, n = 2, p = 0.5)

# 二項乱数
# --- N=10,p=0.5の二項分布に従う乱数
# --- 成功確率が0.2の場合に10回の試行で成功が出る回数
np.random.seed(1)
sp.stats.binom.rvs(n = 10, p = 0.2, size = 5)

# 二項分布
# --- N=10,p=0.2の二項分布
# --- オブジェクトのみ生成
binomial = sp.stats.binom(n = 10, p = 0.2)

# 二項乱数
np.random.seed(1)
rvs_binomial = binomial.rvs(size = 10000)

# 確率質量関数
m = np.arange(0,10,1)
pmf_binomial = binomial.pmf(k = m)

# 乱数のヒストグラムと確率質量関数を重ねる
sns.displot(rvs_binomial, bins = m, kde = False, color = 'gray')
plt.plot(m, pmf_binomial, color = 'black')
plt.show()


# 2 ポアソン分布 -------------------------------------------------------------

# ＜ポイント＞
# - 離散値しかとらないカウントデータの従う分布をポアソン分布という
#   --- ポアソン分布は強度(λ)のみでコントロールされる


# 確率質量関数
# --- k=強度(λ)
sp.stats.poisson.pmf(k = 2, mu = 5)

# ポアソン乱数
# --- λ=2のポアソン分布に従う乱数
np.random.seed(1)
sp.stats.poisson.rvs(mu = 2, size = 5)

# ポアソン分布
# --- λ=2のポアソン分布
poisson = sp.stats.poisson(mu = 2)

# 乱数
np.random.seed(1)
rvs_poisson = poisson.rvs(size = 10000)

# 確率質量関数
pmf_poisson = poisson.pmf(k = m)

# 乱数のヒストグラムと確率質量関数を重ねる
sns.displot(rvs_poisson, bins = m, kde = False, color = 'gray')
plt.plot(m, pmf_poisson, color = 'black')
plt.show()


# 3 二項分布とポアソン分布の関係 --------------------------------------------------

# ポアソン分布は二項分布から導き出すことができる
#   --- 成功確率pがとても小さい二項分布がポアソン分布
#   --- ポアソン分布は故障/交通事故などめったにない事象を扱う


# Nが大きくpが小さい二項分布
N = 100000000
p = 0.00000002
binomial_2 = sp.stats.binom(n = N, p = p)

# 確率質量関数
pmf_binomial_2 = binomial_2.pmf(k = m)

# 確率質量のグラフ
plt.plot(m, pmf_poisson, color = 'gray')
plt.plot(m, pmf_binomial_2, color = 'black', linestyle = 'dotted')
plt.show()
