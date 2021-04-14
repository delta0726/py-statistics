# ***************************************************************************************
# Title     : あたらしいPythonで学ぶ統計学の教科書
# Chapter   : 3-1 Pythonによる記述統計：1変量データ編
# Created by: Owner
# Created on: 2021/4/15
# Page      : P106 - P115
# ***************************************************************************************


# ＜概要＞
# - 1変量データの統計量の算出方法を確認


# ＜ポイント＞
# - 複数のライブラリで同じ処理を関数やメソッドとして実装しているケースがある
# - 本書ではScipyを優先して使用する


# ＜目次＞
# 0 準備
# 1 合計/平均
# 2 分散/標準偏差
# 3 標準化
# 4 その他の統計量
# 5 scipy.statsと四分位点


# 0 準備 -------------------------------------------------------------------------------

import numpy as np
import scipy as sp

from scipy import stats


# 1変量データの管理
fish_data = np.array([2, 3, 3, 4, 4, 4, 4, 5, 5, 6])
fish_data_2 = np.array([2, 3, 3, 4, 4, 4, 4, 5, 5, 100])
fish_data_3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])


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


# 3 標準化 ---------------------------------------------------------------------------------

# 計算部品
mu = sp.sum(fish_data) / N
sigma = sp.std(fish_data, ddof=1)

# 標準化
# --- 計算プロセス
zscore = (fish_data - mu) / sigma
zscore


# 4 その他の統計量 --------------------------------------------------------------------------

# 最大値
sp.amax(fish_data)

# 最小値
sp.amin(fish_data)

# 中央値
sp.median(fish_data)

# 中央値のロバスト性
# --- 異常値を含むデータ
# --- 平均値は異常値の影響を強く受ける
# --- 中央値はロバスト
sp.mean(fish_data_2)
sp.median(fish_data_2)


# 5 scipy.statsと四分位点 ------------------------------------------------------------------

# ＜ポイント＞
# - scipy.statsには統計分析に特化した関数が準備されている
#   --- 以下ではパーセンタイルの値を取得する関数を確認する

# パーセンタイルの値を取得
stats.scoreatpercentile(fish_data_3, 25)
stats.scoreatpercentile(fish_data_3, 75)

# 参考：ライブラリ構成
dir(stats)
