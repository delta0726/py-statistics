# ***************************************************************************************
# Title     : あたらしいPythonで学ぶ統計学の教科書
# Chapter   : 3-1 Pythonによる記述統計：1変量データ編
# Created by: Owner
# Created on: 2021/4/15
# Page      : P106 - P115
# ***************************************************************************************


# ＜概要＞


# ＜ポイント＞
# - 複数のライブラリで同じ処理を関数やメソッドとして実装しているケースがある
# - 本書ではScipyを優先して使用する


# ＜目次＞
# 0 準備


# 0 準備 -------------------------------------------------------------------------------

import numpy as np
import scipy as sp

# 1変量データの管理
fish_data = np.array([2, 3, 3, 4, 4, 4, 4, 5, 5, 6])


# 1 合計/平均 -------------------------------------------------------------------------

# 合計
sp.sum(fish_data)
np.sum(fish_data)
sum(fish_data)
fish_data.sum()

# サンプルサイズ
len(fish_data)

# 平均（期待値）
# --- 計算プロセス
N = len(fish_data)
sum_value = sp.sum(fish_data)
mu = sum_value / N
mu

# 平均（期待値）
# --- 関数による実装
sp.mean(fish_data)


# 2 分散/標準偏差 -------------------------------------------------------------------------

# 計算部品
mu = sp.sum(fish_data) / N

# 標本分散
# --- 計算プロセス
sigma_2_sample = sp.sum((fish_data - mu) ** 2) / N
sigma_2_sample

# 標本分散
# --- 関数による実装
# --- 標本分散の際はddofを0とする
sp.var(fish_data, ddof=0)

# 不偏分散
# --- 標本分散の際はddofを1とする
sigma_2 = sp.sum((fish_data - mu) ** 2) / (N - 1)
sigma_2

# 標本分散
# --- 関数による実装
# --- 標本分散の際はddofを1とする
sp.var(fish_data, ddof=1)


# 標準偏差
# --- 不偏分散から計算する
sigma = sp.sqrt(sigma_2)
sigma

# 標準偏差
# --- 関数による実装
sp.std(fish_data, ddof=1)
