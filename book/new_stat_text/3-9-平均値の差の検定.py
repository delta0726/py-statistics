# ***************************************************************************************
# Title     : あたらしいPythonで学ぶ統計学の教科書
# Chapter   : 3-9 平均値の差の検定
# Created by: Owner
# Created on: 2021/4/23
# Page      : P212 - P219
# ***************************************************************************************


# ＜概要＞
# - 2つの変数の間で平均値に差があるかどうかを判断する


# ＜目次＞
# 0 準備
# 1 対応のあるt検定
# 2 対応のないt検定


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
import scipy as sp

from scipy import stats


# データ準備
paired_test_data = pd.read_csv("book/new_stat_text/csv/3-9-1-paired-t-test.csv")
paired_test_data


# 1 対応のあるt検定 ----------------------------------------------------------------------

# ＜ポイント＞
# - 薬を飲む前と後における体温の差を計算する
#   --- 前後の体温の差が0と異なるかどうかを検定で調べる

# データ抽出
before = paired_test_data.query('medicine == "before"')["body_temperature"]
after = paired_test_data.query('medicine == "after"')["body_temperature"]

# 配列に変換
before = np.array(before)
after = np.array(after)

# 差を計算
diff = after - before
diff

# t検定
# --- 差の値が平均0と異なるか？
# --- 1群のt検定で調べる
stats.ttest_1samp(a=diff, popmean=0)
stats.ttest_rel(a=after, b=before)


# 2 対応のないt検定 ----------------------------------------------------------------------

# ＜ポイント＞
# - 薬を飲む前と後における体温の差を計算する
#   --- 前後の体温の差が0と異なるかどうかを検定で調べる

# データ抽出
before = paired_test_data.query('medicine == "before"')["body_temperature"]
after = paired_test_data.query('medicine == "after"')["body_temperature"]

# 平均値
mean_bef = sp.mean(before)
mean_aft = sp.mean(after)

# 分散
sigma_bef = sp.var(before, ddof=1)
sigma_aft = sp.var(after, ddof=1)

# サンプルサイズ
m = len(before)
n = len(after)

# t値
t_value = (mean_aft - mean_bef) / sp.sqrt((sigma_bef / m + sigma_aft / n))
t_value

# t値
# --- 関数
stats.ttest_ind(a=after, b=before, equal_var=False)
