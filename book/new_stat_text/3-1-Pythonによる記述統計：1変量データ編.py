# ***********************************************************************************************
# Title   : あたらしいPythonで学ぶ統計学の教科書
# Chapter : 3 Pythonによるデータ分析
# Theme   : 1 Pythonによる記述統計：1変量データ編
# Date    : 2022/05/03
# Page    : P106 - P115
# URL     : https://logics-of-blue.com/python-stats-book-support/
# ***********************************************************************************************


# ＜概要＞
# - 1変量データの統計量の算出方法を確認


# ＜メモ＞
# - 書籍はscipyで記述しているが、リタイア警告が出るためnumpyで記述
#   --- sum/mean/sqrtなど当ページのほぼ全ての関数で警告が出現
#   --- scipy.sum is deprecated and will be removed in SciPy 2.0.0, use numpy.sum instead


# ＜目次＞
# 0 準備
# 1 合計/平均
# 2 分散/標準偏差
# 3 標準化
# 4 その他の統計量
# 5 scipy.statsと四分位点


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import numpy as np
import scipy as sp

from scipy import stats


# データ定義
# --- 1変量データの管理
fish_data = np.array([2, 3, 3, 4, 4, 4, 4, 5, 5, 6])
fish_data_2 = np.array([2, 3, 3, 4, 4, 4, 4, 5, 5, 100])
fish_data_3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])


# 1 合計/平均 -------------------------------------------------------------------------

# ＜ポイント＞
# - 複数のライブラリで同じ処理を関数やメソッドとして実装しているケースがある
# - 本書ではScipyを優先して使用する
# - サンプルサイズを求める関数はPytonの標準関数を使用


# 合計
np.sum(fish_data)
sum(fish_data)
fish_data.sum()

# サンプルサイズ
len(fish_data)

# 平均（期待値）
# --- 計算プロセス
N = len(fish_data)
sum_value = np.sum(fish_data)
mu = sum_value / N
mu

# 平均（期待値）
# --- 関数による実装
np.mean(fish_data)


# 2 分散/標準偏差 -------------------------------------------------------------------------

# ＜ポイント＞
# - 標本分散はNで除するが、不偏分散はN-1で除して計算する
#   --- 標本分散は分散を過小に見積もるバイアスを持つ
#   --- 不偏分散はN-1で割るので標本分散より少し大きくなる


# 平均値
# --- 分散を計算するためのパーツ
mu = np.sum(fish_data) / N

# 標本分散
# --- 公式に基づいて計算
sigma_2_sample = np.sum((fish_data - mu) ** 2) / N
sigma_2_sample

# 標本分散
# --- 関数による計算
np.var(fish_data, ddof=0)


# 不偏分散
# --- 公式に基づいて計算）
sigma_2 = np.sum((fish_data - mu) ** 2) / (N - 1)
sigma_2

# 標本分散
# --- 関数による計算
np.var(fish_data, ddof=1)


# (不偏)標準偏差
# --- 不偏分散から計算する
sigma = np.sqrt(sigma_2)
sigma

# (不偏)標準偏差
# --- 関数による実装
np.std(fish_data, ddof=1)


# 3 標準化 ---------------------------------------------------------------------------------

# ＜ポイント＞
# - 標準化はデータ系列を平均を0で標準偏差を1にする変換のこと


# 計算パーツ
# --- 平均値
# --- (不偏)標準偏差
mu = np.sum(fish_data) / N
sigma = np.std(fish_data, ddof=1)

# 標準化
# --- 計算プロセス
zscore = (fish_data - mu) / sigma
zscore


# 4 その他の統計量 --------------------------------------------------------------------------

# 最大値
np.amax(fish_data)

# 最小値
np.amin(fish_data)

# 中央値
np.median(fish_data)

# 中央値のロバスト性
# --- 異常値を含むデータ
# --- 平均値は異常値の影響を強く受ける
# --- 中央値はロバスト
np.mean(fish_data_2)
np.median(fish_data_2)


# 5 scipy.statsと四分位点 ------------------------------------------------------------------

# ＜ポイント＞
# - scipy.statsには統計分析に特化した関数が準備されている
#   --- 以下ではパーセンタイルの値を取得する関数を確認する

# パーセンタイルの値を取得
stats.scoreatpercentile(fish_data_3, 25)
stats.scoreatpercentile(fish_data_3, 75)
