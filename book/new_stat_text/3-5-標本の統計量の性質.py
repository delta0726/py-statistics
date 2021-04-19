# ***************************************************************************************
# Title     : あたらしいPythonで学ぶ統計学の教科書
# Chapter   : 3-5 標本の統計量の性質
# Created by: Owner
# Created on: 2021/4/19
# Page      : P157 - P177
# ***************************************************************************************


# ＜概要＞
# - サンプリングは通常は1度霧しかできない
# - シミュレーションを用いることで、サンプリングを何度も繰り返すことができる


# ＜目次＞
# 0 準備
# 1 標本平均を何度も計算してみる
# 2 標本平均の平均値は母平均に近い
# 3 サンプルサイズ大なら標本平均は母平均に近い
# 4 標本平均を何度も作成する関数を定義する
# 5 サンプルサイズを変えたときに標本平均の分布
# 6 標本平均の標準偏差は母標準偏差よりも小さい
# 7 標準誤差
# 8 標本分散の平均値は母分散からずれている
# 9 不偏分散を使うとバイアスがなくなる
# 10 サンプルサイズ大なら不偏分散は母分散に近い


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

# 母集団の準備
# --- ｢平均4、標準偏差0.8｣の正規分布を母集団と仮定する
population = stats.norm(loc=4, scale=0.8)


# 1 標本平均を何度も計算してみる ---------------------------------------------------------

# ＜ポイント＞
# - 標本平均を作成するシミュレーションを10000回だけ実行する
#   --- 標本平均は母集団から10個の値を抽出して計算する


# シミュレーション回数
x = 10000

# 配列準備
# --- シミュレーション結果を格納
sample_mean_array = np.zeros(x)

# シミュレーション実行
# --- 母集団から10個の数値を抽出(平均4、標準偏差0.8の正規分布)
# --- 小数を含む
np.random.seed(1)
for i in range(x):
    sample = population.rvs(size=10)
    sample_mean_array[i] = sp.mean(sample)

# 結果確認
sample_mean_array


# 2 標本平均の平均値は母平均に近い -------------------------------------------------------

# ＜ポイント＞
# - ｢サンプルサイズ｣と｢標本平均｣と｢母平均｣の関係を確認する


# 標本平均の平均値
# --- 母平均が4なので、4.004はそれなりに近い値が出力された
sp.mean(sample_mean_array)

# 標本平均の標準偏差
# --- 母標準偏差が0.8なので、0.251はかなり小さい値が出力された
sp.std(sample_mean_array, ddof=1)

# ヒストグラム作成
sns.distplot(sample_mean_array, color='black')
plt.show()


# 3 サンプルサイズ大なら標本平均は母平均に近い --------------------------------------------

# ＜ポイント＞
# - ｢サンプルサイズ｣と｢標本平均｣と｢母平均｣の関係を確認する
# - 標本平均のサンプル数を回数ごとに増やして標本平均を作成していく
#   --- サンプル数が増えることで標本平均が母平均に収斂することを確認


# サンプルサイズの作成
# --- 10-100010まで100区切りで変化
size_array = np.arange(start=10, stop=100100, step=100)
size_array

# 配列準備
# --- シミュレーション結果を格納
sample_mean_array_size = np.zeros(len(size_array))
sample_mean_array_size

# 配列数の確認
len(size_array)
len(sample_mean_array_size)

# シミュレーション実行
# --- 標本平均のサンプル数を回数ごとに増やす
np.random.seed(1)
for i in range(0, len(size_array)):
    sample = population.rvs(size=size_array[i])
    sample_mean_array_size[i] = sp.mean(sample)

# プロット作成
# --- サンプル数が増えることで標本平均が母平均に収斂していく
plt.plot(size_array, sample_mean_array_size, color='black')
plt.xlabel("sample size")
plt.xlabel("sample mean")
plt.show()


# 4 標本平均を何度も作成する関数を定義する --------------------------------------------------

# ＜ポイント＞
# - 3のプロセスを関数化
#   --- 関数内で外部参照の警告がでるが今回は無視


# 関数定義
def calc_sample_mean(size, n_trial):
    sample_mean_array = np.zeros(n_trial)
    for i in range(0, n_trial):
        sample = population.rvs(size=size)
        sample_mean_array[i] = sp.mean(sample)
    return(sample_mean_array)


# 動作確認
np.random.seed(1)
sp.mean(calc_sample_mean(size=10, n_trial=10000))


# 5 サンプルサイズを変えたときに標本平均の分布 -----------------------------------------------

# ＜ポイント＞
# - サンプルサイズが多くなると標本平均のバラツキが小さくなることを再度確認する


# シード設定
np.random.seed(1)

# サンプルサイズ：10
size_10 = calc_sample_mean(size=10, n_trial=10000)
size_10_df = pd.DataFrame({"sample_mean": size_10,
                           "size": np.tile("size 10", 10000)})

# サンプルサイズ：10
size_20 = calc_sample_mean(size=20, n_trial=10000)
size_20_df = pd.DataFrame({"sample_mean": size_20,
                           "size": np.tile("size 20", 10000)})

# サンプルサイズ：30
size_30 = calc_sample_mean(size=30, n_trial=10000)
size_30_df = pd.DataFrame({"sample_mean": size_30,
                           "size": np.tile("size 30", 10000)})

# データ統合
sim_result = pd.concat([size_10_df, size_20_df, size_30_df])
sim_result

# プロット比較
sns.violinplot(x="size", y="sample_mean", data=sim_result, color="gray")
plt.show()


# 6 標本平均の標準偏差は母標準偏差よりも小さい ------------------------------------------

# ＜ポイント＞
# - サンプルサイズが小さいとサンプル平均のバラツキが大きくなる
#   --- 標本平均の標準偏差をサンプルサイズごとに見ていくことで確認

# データ作成
# --- サンプルサイズ
size_array = np.arange(start=2, stop=102, step=2)
size_array

# ゼロベクトル
# --- 標本平均の格納用
sample_mean_std_array = np.zeros(len(size_array))

# シミュレーション
# --- サンプルサイズを変えて標本平均を算出(100回)
# --- 標本平均100個の標本標準偏差を算出
# --- 上記の試行を50回行う
np.random.seed(1)
for i in range(0, len(size_array)):
    sample_mean = calc_sample_mean(size=size_array[i], n_trial=100)
    sample_mean_std_array[i] = sp.std(sample_mean, ddof=1)

# プロット作成
plt.plot(size_array, sample_mean_std_array, color="black")
plt.xlabel("sample size")
plt.ylabel("mean_std value")
plt.show()


# 7 標準誤差 ---------------------------------------------------------------------

# ＜ポイント＞
# - 標準誤差とは、理論上の標本平均の標準偏差のことを指す
# - 標準誤差は母標準偏差よりも必ず小さくなる
#   --- 数式より自明（直感的な説明はP170）

# 標準誤差の算出
# --- 理論上の標本平均の標準偏差
# --- 0.8は母集団の標準偏差
standard_error = 0.8 / np.sqrt(size_array)
standard_error

# プロット比較
# --- 実線：シミュレーション
# --- 点線：標準誤差(理論上の標本平均の標準偏差)
plt.plot(size_array, sample_mean_std_array, color="black")
plt.plot(size_array, standard_error, color="black", linestyle="dotted")
plt.xlabel("sample size")
plt.ylabel("mean_std value")
plt.show()


# 8 標本分散の平均値は母分散からずれている -------------------------------------------

# ＜ポイント＞
# - 標本分散の平均値は母分散よりも小さくなる(過小評価される)

# ベクトル生成
# --- 標本分散の格納用
sample_var_array = np.zeros(10000)

# シミュレーション
# --- データを10個選んで標本分散を求める(10000回)
# --- ddof=0： 標本分散
np.random.seed(1)
for i in range(0, 10000):
    sample = population.rvs(size=10)
    sample_var_array[i] = sp.var(sample, ddof=0)

# 標本分散の標本平均
# --- 期待値は0.64 (0.8^2)
sp.mean(sample_var_array)


# 9 不偏分散を使うとバイアスがなくなる -----------------------------------------------

# ＜ポイント＞
# - 不偏分散の平均値は母分散とみなすことができる
#   --- 不偏分散を使うとバイアスがなくなる

# ベクトル生成
# --- 不偏分散の格納用
unbias_var_array = np.zeros(10000)

# シミュレーション
# --- データを10個選んで標本分散を求める(10000回)
# --- ddof=1： 不偏分散
np.random.seed(1)
for i in range(0, 10000):
    sample = population.rvs(size=10)
    unbias_var_array[i] = sp.var(sample, ddof=1)

# 標本分散の標本平均
# --- 期待値は0.64 (0.8^2)
sp.mean(unbias_var_array)


# 10 サンプルサイズ大なら不偏分散は母分散に近い ------------------------------------

# ＜ポイント＞
# - 不偏分散の平均値は母分散とみなすことができる

# ベクトル生成
# --- サンプルサイズの格納
size_array = np.arange(start=10, stop=100100, step=100)
size_array

# ベクトル生成
# --- 不偏分散の格納用
unbias_var_array_size = np.zeros(len(size_array))

# シミュレーション
np.random.seed(1)
for i in range(0, len(size_array)):
    sample = population.rvs(size=size_array[i])
    unbias_var_array_size[i] = sp.var(sample, ddof=1)

# 不偏分散の標本平均
# --- 期待値は0.64 (0.8^2)
sp.mean(unbias_var_array_size)

# プロット作成
plt.plot(size_array, unbias_var_array_size, color="black")
plt.xlabel("sample size")
plt.ylabel("unbias var")
plt.show()


# 11 中心極限定理 ---------------------------------------------------------------

# ＜ポイント＞
# - 母集団分布が何であっても、サンプルサイズが大きいときには確率変数の和は正規分布に近づく
#   --- 中心極限定理

# ＜事例＞
# - コインを投げると裏/表は二項分布に従う

# パラメータ設定
# --- サンプルサイズ
# --- 試行回数
n_size = 10000
n_trial = 50000

# コインの定義
# --- 表： 1
# --- 裏： 0
coin = np.array([0, 1])

# 表が出た回数
count_coin = np.zeros(n_trial)

# シミュレーション
# --- コインをn_size回投げる試行をn_trial回行う
np.random.seed(1)
for i in range(0, n_trial):
    count_coin[i] = sp.sum(np.random.choice(coin, size=n_size, replace=True))

# プロット作成
# --- ヒストグラム
sns.distplot(count_coin, color="black")
plt.show()
