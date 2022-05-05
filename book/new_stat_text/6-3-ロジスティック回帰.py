# ***********************************************************************************************
# Title   : あたらしいPythonで学ぶ統計学の教科書
# Chapter : 6 一般化線形モデル
# Theme   : 3 ロジスティック回帰
# Date    : 2022/05/00
# Page    : P356 - P369
# URL     : https://logics-of-blue.com/python-stats-book-support/
# ***********************************************************************************************


# ＜概要＞
# - 一般化線形モデルの代表例としてロジスティック回帰を確認する
#   --- 確率分布に二項分布、リンク式にロジット関数を用いている
#   --- 線形予測子は正規線形モデルと同じ


# ＜目次＞
# 0 準備
# 1 データ確認
# 2 モデル構築
# 3 モデル選択
# 4 ロジスティクス回帰のプロット
# 5 成功確率の予測
# 6 回帰係数とオッズ比の関係


# 0 準備 -------------------------------------------------------------------------------

# 数値計算に使うライブラリ
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats

from matplotlib import pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm

sns.set()

# データロード
test_result = pd.read_csv("csv/6-3-1-logistic-regression.csv")

# データ確認
test_result

# 1 データ確認 ----------------------------------------------------------------------

# クロス集計
# --- 勉強時間ごとのカウント
pd.crosstab(test_result["hours"], test_result["result"])

# プロット作成
# データの図示
sns.barplot(x="hours", y="result", data=test_result, palette='gray_r')
plt.show()

# グループ集計
# --- 勉強時間ごとの合格率
test_result.groupby("hours").mean()

# 2 モデル構築 -----------------------------------------------------------------------

# モデル定義
# --- 確率分布：family = Binomial()
# --- リンク関数：デフォルト値が自動で指定される
mod_glm = smf.glm(formula="result ~ hours",
                  data=test_result,
                  family=sm.families.Binomial()).fit()

# モデル定義（エラー）
# logistic_reg = smf.glm(formula="result ~ hours",
#                        data=test_result,
#                        family=sm.families.Binomial(link=sm.families.links.logit)).fit()

# 結果の出力
mod_glm.summary()


# 3 モデル選択 ----------------------------------------------------------------------

# ＜ポイント＞
# - hoursモデルとNullモデルをAICで比較する
#   --- hoursモデルの方がAICが小さいのでhoursは予測に役立っていることが分かる


# Nullモデル
mod_glm_null = smf.glm(formula="result ~ 1",
                       data=test_result,
                       family=sm.families.Binomial()).fit()

# AICの比較
# --- Nullモデル
# --- 変数入りモデル
mod_glm_null.aic
mod_glm.aic


# 4 ロジスティク回帰のプロット ------------------------------------------------------

# ＜ポイント＞
# - 合否サンプルの散布図に理論上の合格率を重ねて作成する
#   --- 散布図は点が重なるのでラグプロットにしておく


# プロット作成
# --- lmplotを用いて回帰プロットを作成
# --- logistic=Trueとすることでロジスティック回帰曲線を図示
sns.lmplot(x="hours", y="result",
           data=test_result,
           logistic=True,
           scatter_kws={"color": "black"},
           line_kws={"color": "black"},
           x_jitter=0.1, y_jitter=0.02)

plt.show()


# 5 成功確率の予測 --------------------------------------------------------------

# データ作成
# 0~9まで1ずつ増える等差数列
exp_val = pd.DataFrame({"hours": np.arange(0, 10, 1)})

# 成功確率の予測値
pred = mod_glm.predict(exp_val)
pred


# 6 回帰係数とオッズ比の関係 --------------------------------------------------------

# ＜ポイント＞
# - 回帰係数は対数オッズ比と一致する
#   --- 回帰係数は説明変数を1単位変化させたときの対数オッズ比となる
#   --- 回帰係数をexpで変換することで、説明変数が1単位増えたときにオッズが何倍になるかを示す


# 勉強時間が1時間の時の合格率
exp_val_1 = pd.DataFrame({"hours": [1]})
pred_1 = mod_glm.predict(exp_val_1)
pred_1

# 勉強時間が2時間の時の合格率
exp_val_2 = pd.DataFrame({"hours": [2]})
pred_2 = mod_glm.predict(exp_val_2)
pred_2

# オッズ
odds_1 = pred_1 / (1 - pred_1)
odds_2 = pred_2 / (1 - pred_2)

# 対数オッズ比
np.log(odds_2 / odds_1)

# 係数
mod_glm.params["hours"]

# 補足：オッズ比に戻す
# --- 説明変数が1単位増えたときにオッズが何倍になるかを示す
np.exp(mod_glm.params["hours"])
