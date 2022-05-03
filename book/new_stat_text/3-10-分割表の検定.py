# **************************************************************************************
# Title   : あたらしいPythonで学ぶ統計学の教科書
# Chapter : 3 Pythonによるデータ分析
# Theme   : 10 分割表の検定
# Date    : 2022/05/04
# Page    : P220 - P232
# URL     : https://logics-of-blue.com/python-stats-book-support/
# **************************************************************************************


# ＜概要＞
# - 分割表に対する独立性の検定を行う
# - 分割表に対して正しい知識を持つだけでデータ分析の質の向上が見込める


# ＜目次＞
# 0 準備
# 1 p値の計算
# 2 分割表の検定


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import pandas as pd
from scipy import stats

# データ準備
# --- 分割表のロング型データ
click_data = pd.read_csv("csv/3-10-1-click_data.csv")
click_data


# 1 p値の計算 -------------------------------------------------------------------------

1 - stats.chi2.cdf(x=6.667, df=1)


# 2 分割表の検定 ----------------------------------------------------------------------

# 分割表
cross = pd.pivot_table(data=click_data, values="freq", aggfunc="sum",
                       index="color", columns="click")

# 確認
cross

# 検定
stats.chi2_contingency(observed=cross, correction=False)
