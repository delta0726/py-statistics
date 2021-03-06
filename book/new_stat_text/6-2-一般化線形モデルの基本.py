# ***********************************************************************************************
# Title   : あたらしいPythonで学ぶ統計学の教科書
# Chapter : 6 一般化線形モデル
# Theme   : 2 一般化線形モデルの基本
# Date    : 2022/05/05
# Page    : P351 - P355
# URL     : https://logics-of-blue.com/python-stats-book-support/
# ***********************************************************************************************


# ＜概要＞
# - 一般化線形モデルの構成要素を確認する
#   --- 構成要素を切り替えることでモデルの柔軟性を改善することができる


# ＜構成要素＞
# - 母集団の従う確率分布
# - 線形予測子（フォーミュラ）
# - リンク関数


# ＜確率分布＞
# - 一般化線形モデルでは正規分布以外に二項分布やポアソン分布なども適用することができる


# ＜線形予測子＞
# - 説明変数を線形の関係式で表現したものを線形予測子という
#   --- 目的変数との関係を足し算や掛け算を使って表現する
#   --- 線形予測子をそのまま用いるのではなく、次のリンク関数で適切なものに変換する


# ＜リンク関数＞
# - 目的変数と線形予測子の対応を取るための変換式をリンク式という
#   --- リンク関数と確率分布はセットになることが多い（P354）
#   --- 正規線形モデルは変換を行わないので｢恒等関数｣となる


# ＜パラメータ推定＞
# - 一般化線形モデルは正規分布以外も扱うので最尤法によるパラメータ推定が行われる
#   --- 実データの分布が理論分布に尤も近づくようにパラメータを推定


# ＜検定手法＞
# - Wald検定：サンプル数が多いときに推定値が正規分布に従うことを利用した検定手法
#   --- 一般化線形モデルではt検定を行うことができないのでWald検定で代用する

# - 尤度比検定：モデルの当てはまり具合を比較する手法
#   --- 正規線形モデルの分散分析に対応する検定

# - スコア検定：省略
