# ***************************************************************************************
# Title     : あたらしいPythonで学ぶ統計学の教科書
# Chapter   : 3-7 推定
# Created by: Owner
# Created on: 2021/4/21
# Page      : P190 - P199
# ***************************************************************************************


# ＜概要＞
# - 母集団のパラメータを推定する
#   --- 母集団は本来観測することができないため、推定する必要がある


# ＜目次＞
# 0 準備
# 1 点推定
# 2 区間推定
# 3 信頼区間の幅を決める要素
# 4 区間推定の結果の解釈


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns

from scipy import stats
from matplotlib import pyplot as plt

# 描画セット
sns.set()

# データ準備
# --- 標本データ
# --- 母集団から抽出した一部と想定
fish = pd.read_csv("book/new_stat_text/csv/3-7-1-fish_length.csv")
fish


# 1 点推定 -----------------------------------------------------------------------------

# 母平均の点推定
# --- 標本平均を計算するだけ
mu = sp.mean(fish)
mu

# 母分散の点推定
# --- 標本データの不偏分散
sigma_2 = sp.var(fish, ddof=1)
sigma_2


# 2 区間推定 ------------------------------------------------------------------------------

# ＜ポイント＞
# - 区間推定には｢自由度｣｢標本平均｣｢標準誤差｣の3つが必要となる

# 自由度
df = len(fish) - 1
df

# 標本平均
mu = sp.mean(fish)
mu

# 標準誤差
# --- 1サンプルあたりの分散量
sigma = sp.std(fish, ddof=1)
se = sigma / sp.sqrt(len(fish))

# 信頼区間
# --- 左右2.5％
interval = stats.t.interval(alpha=0.95, df=df, loc=mu, scale=se)
interval

# 97.5％点
t_975 = stats.t.ppf(q=0.975, df=df)

# 下限信頼限界
lower = mu - t_975 * se
lower

# 上限信頼限界
upper = mu + t_975 * se
upper


# 3 信頼区間の幅を決める要素 ---------------------------------------------------------------

# 標準誤差を増やす
# --- 標準誤差を10倍に増やすと（標準誤差が上昇）
# --- 信頼区間が広がる
se2 = (sigma * 10) / sp.sqrt(len(fish))
stats.t.interval(alpha=0.95, df=df, loc=mu, scale=se2)

# サンプルサイズを増やす
# --- サンプルサイズを増やすと安定感が増す
# --- 信頼区間が狭くなる
df2 = (len(fish) * 10) - 1
se3 = sigma / sp.sqrt(len(fish) * 10)
stats.t.interval(alpha=0.95, df=df2, loc=mu, scale=se3)

# アルファを増やす
# --- 信頼区間が広がる
stats.t.interval(alpha=0.99, df=df, loc=mu, scale=se)


# 4 区間推定の結果の解釈 -------------------------------------------------------------------

# 配列作成
# --- 信頼区間を格納する入れ物
be_included_array = np.zeros(20000, dtype="bool")
be_included_array

# シミュレーション実行
# --- 母平均：4
# --- 母標準偏差：0.8
np.random.seed(1)
norm_dist = stats.norm(loc=4, scale=0.8)
for i in range(0, 20000):
    sample = norm_dist.rvs(size=10)
    df = len(sample) - 1
    mu = sp.mean(sample)
    std = sp.std(sample, ddof=1)
    se = std / sp.sqrt(len(sample))
    interval = stats.t.interval(alpha=0.95, df=df, loc=mu, scale=se)
    if interval[0] <= 4 <= interval[1]:
        be_included_array[i] = True

# 信頼区間内の割合
sum(be_included_array) / len(be_included_array)
