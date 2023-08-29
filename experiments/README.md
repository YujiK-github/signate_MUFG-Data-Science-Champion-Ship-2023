| file | CV | LB | description | 
| - | - | - | - |
| exp001 | 0.649981155994107 | 0.6579048 | baseline with lgb<br>"is_fraud?"のStratifiedKFold(n_splits=5) |
| exp002 | 0.6514425113080011 | | baseline with lgb & polars<br>["is_fraud?", "card_id", "user_id"]のMultilabelStratifiedKFold(n_splits=5) |
| exp003 | 0.6532026768642447 | 0.6588847 | lgb add features<br>["is_fraud?", "card_id", "user_id"]のMultilabelStratifiedKFold(n_splits=5) |
| exp004 | 0.6561070868244484 (threshold: 0.325) | 0.6581321 | exp003のn_splits増やした<br>["is_fraud?", "card_id", "user_id"]のMultilabelStratifiedKFold(n_splits=10) |
| exp005 | 0.6606264292148846 (threshold: 0.35000000000000003) | 0.6691146 | それぞれのuser_idごとにモデルを作成した<br>["is_fraud?", "card_id"]のMultilabelStratifiedKFold(n_splits=5) |

TODO: 回帰モデルとして解く  
TODO: user_idごとのモデル

TODO: EDA中に考えついた特徴量たちの検証
* merchant_id, merchant_city, merchant_state, mccごとの集約特徴量: その街、州の危険度を表現できるかも
* stateとmccを絡めた特徴量
* カテゴリー変数を絡めた特徴量
  * stateとmcc
  * mccとuse_chip
  * 使用カードとmcc
  * 口座開設と有効期限の差, pinの変更
  * 現在の年齢と退職年齢
  * street_nameの取得
  * 年収関係の差
  * 欠損値の処理