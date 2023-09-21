## 解法の概要

### Cross Validationの設計
* 目的変数はバイナリー
* cardとuserはtrainとtestで共通している
* 他の特徴量はtrainとtestで共通しているもののあれば、それぞれのデータセットに特有のものもあった。

以上より、MultilabelStratifiedKFold.split(X=df, y=df
[["is_fraud?", "user_id", "card_id"]])を行った。

### データの前処理
* "amount", "total_debt", "credit_limit", "yearly_income_person", "per_capita_income_zipcode"はstringだったが、floatにした方が良いと判断したのでfloat化した
* "expires", "acct_open_date"はdatetimeにした

### 特徴量エンジニアリング
今回は目的変数とそれぞれの特徴量の関係性を見るのではなく、2つの特徴量と目的変数の関係を捉えて特徴量を作成しようとした。時間が無かった。

### モデル
ユーザーごとにモデルを作成すると精度が上がった。

### アンサンブル
それぞれの値が[0, 1]であるのでほとんどアンサンブルによる精度の向上は無かった。昔参加したコンペで、sigmoidを通す前にアンサンブルをした方が精度が向上した実験を思い出し、sigmoidのinverse functionを通してアンサンブルしてみたが、効果が無かった。

## 個人的な反省
* userごとにモデルを作成した後にもgreedy forward selection(計算量を落としたver.)を行うべきだった。
* n_splitsを増やしてCVが向上してもLBが向上しなければ過学習にしていると判断するべきだった。[Feedback3でChrisさんがそれぞれのfoldが157程度になるまで細分化していた](https://www.kaggle.com/code/cdeotte/rapids-svr-cv-0-450-lb-0-44x)のを信じ過ぎていた。結局最適なn_splitsの数が分からない。
* polarsの再現性が取れなかった問題。
* n_splitsを早い段階で増やしてしまったので実験に時間がかかった。まだ早い。