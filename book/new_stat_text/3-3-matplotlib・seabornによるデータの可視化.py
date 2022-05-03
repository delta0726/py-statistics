# **************************************************************************************
# Title   : あたらしいPythonで学ぶ統計学の教科書
# Chapter : 3 Pythonによるデータ分析
# Theme   : 3 matplotlib・seabornによるデータの可視化
# Date    : 2022/05/03
# Page    : P129 - P144
# URL     : https://logics-of-blue.com/python-stats-book-support/
# **************************************************************************************


# ＜概要＞
# - matplotlibはPythonの標準的な描画ライブラリ
# - Seabornはmatplotlibをより美しく描くためのライブラリ
#   --- 簡単にmatplotlibの使い方を学んだ後は、Seabornを使って可視化していく


# ＜目次＞
# 0 準備
# 1 pyplotによる折れ線グラフ
# 2 seaborn + pyplotによる折れ線グラフ
# 3 seabornによるヒストグラム
# 4 カーネル密度推定によるヒストグラムの平滑化
# 5 2変量データに対するヒストグラム
# 6 多変量データを図示するコードの書き方
# 7 箱ひげ図
# 8 バイオリンプロット
# 9 棒グラフ
# 10 散布図
# 11 ペアプロット


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt


# データ準備
fish_multi = pd.read_csv("csv/3-3-2-fish_multi_2.csv")
cov_data = pd.read_csv("csv/3-2-3-cov.csv")

# データロード
iris = sns.load_dataset("iris")


# 1 pyplotによる折れ線グラフ --------------------------------------------------------------

# ＜ポイント＞
# - matplotlibはシンプルな(少しそっけない)描画が作成される
# - オブジェクトの重ね書きによってプロットを作成していく
#   --- plt.*が並んでいて冗長なコードになる


# データ作成
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([2, 3, 4, 3, 5, 4, 6, 7, 4, 8])

# プロット作成
plt.plot(x, y, color='black')
plt.title("Lineplot maplotlib")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# 2 seaborn + pyplotによる折れ線グラフ ---------------------------------------------------

# ＜ポイント＞
# - seabornを組み合わせることでプロットのデザインをモダンにすることができる


# データ作成
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([2, 3, 4, 3, 5, 4, 6, 7, 4, 8])

# デザイン設定
sns.set()

# プロット作成
plt.plot(x, y, color='black')
plt.title("Lineplot maplotlib")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# 3 seabornによるヒストグラム ---------------------------------------------------------

# ＜ポイント＞
# - seabornを組み合わせることでプロットのデザインをモダンにすることができる


# データ作成
fish_data = np.array([2, 3, 3, 4, 4, 4, 4, 5, 5, 6])

# ヒストグラム作成
# --- kde引数をFalseとすることでカーネル密度推定が含まれなくなる
sns.distplot(fish_data, bins=5, color='black', kde=False)
plt.show()


# 4 カーネル密度推定によるヒストグラムの平滑化 --------------------------------------------

# ＜ポイント＞
# - ヒストグラムはbinの数を変更するとプロットの印象が大きく変わってしまう
#   --- 分析者は適切なbinの数を指定しなければならない
#   --- カーネル密度推定はbinを指定する必要がない

# ヒストグラム作成
# --- bins=1とするとデータの特徴が全くわからない
sns.displot(fish_data, bins=1, color='black', kde=False)
plt.show()

# ヒストグラム作成
# --- binsは指定せずデフォルトを使う
# --- kdeをTrue(デフォルト)に指定する
# --- X軸が頻度ではなく密度に変更されている(norm_hist引数)
sns.displot(fish_data, color='black', kde=True)
plt.show()


# 5 2変量データに対するヒストグラム ------------------------------------------------------

# データ確認
fish_multi

# 統計量の算出
fish_multi.groupby('species').describe()

# データ格納
length_a = fish_multi.query('species == "A"')["length"]
length_b = fish_multi.query('species == "B"')["length"]

# ヒストグラムの重ね書き
sns.displot(length_a, bins=5, color='black', kde=False)
sns.displot(length_b, bins=5, color='black', kde=False)
plt.show()


# 6 多変量データを図示するコードの書き方 ------------------------------------------------

# ＜ポイント＞
# - 2変量のプロットを作成する際は、Seabornは概ね以下の構文を使う
#   --- 関数名だけ変更するとプロットを変更できる

# ＜構文＞
# sns.関数名(x, y, data, ...)


# 7 箱ひげ図 -------------------------------------------------------------------------

# ＜ポイント＞
# - 1変量ごとの分布を示すプロットとして箱ひげ図がある

# データ確認
fish_multi

# プロット作成
sns.boxplot(x="species", y="length", data=fish_multi, color="gray")
plt.show()


# 8 バイオリンプロット ------------------------------------------------------------------

# ＜ポイント＞
# - 1変量ごとの分布を示すプロットとしてバイオリンプロットがある
#   --- 箱ひげ図より分布をよく示している

# データ確認
fish_multi

# プロット作成
sns.violinplot(x="species", y="length", data=fish_multi, color="gray")
plt.show()


# 9 棒グラフ --------------------------------------------------------------------------

# ＜ポイント＞
# - 1変量ごとの分布を示すプロットとしてバイオリンプロットがある
#   --- 箱ひげ図より分布をよく示している

# データ確認
fish_multi

# プロット作成
sns.barplot(x="species", y="length", data=fish_multi, color="gray")
plt.show()


# 10 散布図 ----------------------------------------------------------------------------

# ＜ポイント＞
# - 1変量ごとの分布を示すプロットとしてバイオリンプロットがある
#   --- 箱ひげ図より分布をよく示している

# データ確認
cov_data

# プロット作成
sns.jointplot(x="x", y="y", data=cov_data, color="black")
plt.show()


# 11 ペアプロット -----------------------------------------------------------------------

# ＜ポイント＞
# - データセットの散布図行列を作成する
#   --- 散布図は2変量のみだが、ペアプロットは多変量データを可視化することが可能

# データ確認
iris

# グループ平均
iris.groupby("species").mean()

# プロット作成
sns.pairplot(iris, hue="species", palette="gray")
plt.show()
