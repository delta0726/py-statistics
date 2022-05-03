# **************************************************************************************
# Title   : あたらしいPythonで学ぶ統計学の教科書
# Chapter : 3 Pythonによるデータ分析
# Theme   : 8 統計的仮説検定
# Date    : 2022/05/04
# Page    : P200 - P211
# URL     : https://logics-of-blue.com/python-stats-book-support/
# **************************************************************************************


# ＜概要＞
# - 1変量データのt検定を通して仮説検定の考え方を学ぶ


# ＜目次＞
# 0 準備
# 1 t検定の要領
# 2 t検定の実装：t値の計算
# 3 t検定の実装：p値の計算
# 4 シミュレーションによるp値の計算


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
import scipy as sp

from scipy import stats

# データロード
# --- スナック菓子の重量を測定した架空データ
junk_food = pd.read_csv("csv/3-8-1-junk-food-weight.csv")
junk_food


# 1 t検定の要領 -----------------------------------------------------------------------

# ＜設定＞
# 帰無仮説： スナック菓子の平均重量は50gである
# 対立仮設： スナック菓子の平均重量は50gではない
# 有意水準： 5%

# ＜ソリューション＞
# p値が0.05を下回れば帰無仮説は棄却され、スナック菓子の重要は50gと異なるという主張が可能となる


# 2 t検定の実装：t値の計算 -------------------------------------------------------------

# ＜ポイント＞
# - t値とは、平均値と目標値が標準誤差でどれくらい乖離しているかを計算したもの
# - t_value = (mu - x) / se

# 標本平均
mu = sp.mean(junk_food)
mu

# 自由度
df = len(junk_food) - 1
df

# 標本標準偏差
sigma = sp.std(junk_food, ddof=1)
sigma

# 標準誤差
se = sigma / sp.sqrt(len(junk_food))
se

# t値
# --- 理論値
t_value = (mu - 50) / se
t_value


# 3 t検定の実装：p値の計算 --------------------------------------------------------------

# p値
alpha = stats.t.cdf(t_value, df = df)
(1 - alpha) * 2

# 関数によるt検定
# --- t値とp値が直接出力される
stats.ttest_1samp(junk_food, 50)


# 4 シミュレーションによるp値の計算 ------------------------------------------------------

# t値の確認
# --- ボーダー
t_value

# パラメータ設定
size = len(junk_food)
sigma = sp.std(junk_food, ddof=1)
mu = 50

# 配列作成
# --- t値の格納用
t_value_array = np.zeros(50000)

# シミュレーション
np.random.seed(1)
norm_dist = stats.norm(loc=mu, scale=sigma)
for i in range(0, 50000):
    sample = norm_dist.rvs(size=size)
    sample_mean = sp.mean(sample)
    sample_std = sp.std(sample, ddof=1)
    sample_se = sample_std / sp.sqrt(size)
    t_value_array[i] = (sample_mean - mu) / sample_se

# p値
# --- シミュレーションで算出したt値がボーダーを上回った割合
# --- 理論上の値とほぼ一致
(sum(t_value_array > np.array(t_value)) / 50000) * 2
