# **************************************************************************************
# Title   : あたらしいPythonで学ぶ統計学の教科書
# Chapter : 3 Pythonによるデータ分析
# Theme   : 2 Pythonによる記述統計：多変量データ編
# Date    : 2022/05/03
# Page    : P116 - P128
# URL     : https://logics-of-blue.com/python-stats-book-support/
# **************************************************************************************


# ＜概要＞
# - 多変量データの統計量の算出方法を確認
# - 多変量データを扱う際には整然データの概念が重要となることを確認する（3-2-1参照）


# ＜目次＞
# 0 準備
# 1 グループ別の統計量
# 2 クロス集計表
# 3 共分散
# 4 分散共分散行列
# 5 ピアソンの積率相関係数


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import pandas as pd
import numpy as np


# データ準備
# --- 1 グループ別の統計量
fish_multi = pd.read_csv("csv/3-2-1-fish_multi.csv")
fish_multi

# データ準備
# --- 2 クロス集計表
shoes = pd.read_csv("csv/3-2-2-shoes.csv")
shoes

# データ準備
# --- 3 共分散
cov_data = pd.read_csv("csv/3-2-3-cov.csv")
cov_data


# 1 グループ別の統計量 -------------------------------------------------------------------

# 元データの確認
# --- 整然データ(ロング型)
fish_multi

# グループ化
# --- DataFrameGroupByオブジェクトが生成される
# --- pandas.core.frame.DataFrameオブジェクトではない点に注意
group = fish_multi.groupby("species")
group

# 平均
group.mean()

# 標準偏差
group.std(ddof=1)

# 基本統計量
group.describe()


# 2 クロス集計表 -----------------------------------------------------------------------

# ＜ポイント＞
# - データ管理は整然データで行って必要に応じてクロス集計表に変換する


# 元データの確認
# --- 整然データ(ロング型)
shoes

# クロス集計
cross = pd.pivot_table(data=shoes, values="sales", aggfunc="sum",
                       index="store", columns="color")

# データの確認
# --- 集計データ()
cross


# 3 共分散 ------------------------------------------------------------------------------

# ＜ポイント＞
# - 共分散は2変量の数値データの関係性を見るときに使用される
#   --- 相関係数は共分散を基準化したもの


# 元データの確認
cov_data

# 系列の抽出
x = cov_data["x"]
y = cov_data["y"]

# サンプルサイズ
N = len(cov_data)

# 系列の平均値
mu_x = np.mean(x)
mu_y = np.mean(y)

# 標本共分散
cov_sample = sum((x - mu_x) * (y - mu_y)) / N
cov_sample

# (不偏)共分散
cov = sum((x - mu_x) * (y - mu_y)) / (N - 1)
cov


# 4 分散共分散行列 --------------------------------------------------------------------

# ＜ポイント＞
# - 複数系列の共分散は分散共分散行列として表記する


# 元データの確認
cov_data

# 系列の抽出
x = cov_data["x"]
y = cov_data["y"]

# 分散共分散行列の計算
# --- 母数をNとする
np.cov(x, y, ddof=0)

# 分散共分散行列の計算
# --- 母数をN-1とする
np.cov(x, y, ddof=1)


# 5 ピアソンの積率相関係数 ----------------------------------------------------------------

# ＜ポイント＞
# - 相関係数は線形的な関係性のみを評価できる
#   --- P128のような非線形な関係性は適切に評価できない点に注意

# 元データの確認
cov_data

# 系列の抽出
x = cov_data["x"]
y = cov_data["y"]

# 分散の計算
sigma_2_x = np.var(x, ddof=1)
sigma_2_y = np.var(y, ddof=1)

# 共分散の計算
cov = sum((x - np.mean(x)) * (y - np.mean(y))) / (len(cov_data) - 1)

# 相関係数
# --- 公式に基づいて計算
rho = cov / np.sqrt(sigma_2_x * sigma_2_y)
rho

# 相関係数
# --- 関数で計算
rho = np.corrcoef(x, y)
rho
