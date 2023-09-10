import polars as pl
from tqdm.auto import tqdm
import argparse
from datasets import load_data
from model_selection import kfold
from models import train_lgb, train_xgb
from feature_engineering import apply_fe
from utils import seed_everything, get_score

import warnings
warnings.simplefilter("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--n_splits", default=5, type=int, required=False)
    parser.add_argument("--filename", default="expXXX", type=str, required=False)
    parser.add_argument("--save_dir", default="G:/マイドライブ/signate_MUFJ2023/exp/", type=str, required=False)
    parser.add_argument("--data_dir", default="G:/マイドライブ/signate_MUFJ2023/data/", type=str, required=False)
    parser.add_argument("--debug", action="store_true", required=False)
    parser.add_argument("--target_col", default="is_fraud?", type=str, required=False)
    parser.add_argument("--training_per_user_id", action="store_false", required=False)
    parser.add_argument("--show_log", action="store_true", required=False)
    parser.add_argument("--model_type", default="xgb", type=str, required=False, choices=["lgb", "xgb"])
    parser.add_argument("--categorical_features", 
                        default=[
                            "errors?", 'merchant_id', 'merchant_city','merchant_state','zip',"mcc",'use_chip','card_brand','card_type', 'has_chip','gender', 'city',
                            'state', 'zipcode',"card_id", "user_id","same_zipcode_as_zip","city_is_not_America", 
                            ],
                        type=str, nargs="*", required=False)
    parser.add_argument("--numerical_features",
                        default=[
                            "amount", 'cards_issued', 'credit_limit','year_pin_last_changed','current_age','retirement_age','birth_year','birth_month', 'latitude', 'longitude',
                            'per_capita_income_zipcode', 'yearly_income_person', 'total_debt','fico_score', 'num_credit_cards', 'expires_month','expires_year','acct_open_date_month', 
                            'acct_open_date_year',"NonFraudAvgAmount_per_user_card", "merchant_id_count_encoding", 
                            ],
                        type=str, nargs="*", required=False
                        )
    return parser.parse_args()



def main(args) -> None:
    # fix seed
    seed_everything(args.seed)
    
    # data loading
    all_data = load_data(args)
    
    # apply feature engineering
    all_data = apply_fe(all_data)
    
    # split data
    train = all_data.filter(pl.col("flag") == "train")
    test = all_data.filter(pl.col("flag") == "test")
    
    # apply kfold
    train = kfold(args, train)
    
    # train
    args.use_features = args.numerical_features + [col+"_category" for col in args.categorical_features]
    
    if args.training_per_user_id:
        oof_df, test_df = pl.DataFrame(), pl.DataFrame()
        for user in tqdm(train["user_id"].unique()):
            if args.model_type == "lgb":
                _oof_df, _test_df = train_lgb(
                    args, 
                    train.filter(pl.col("user_id") == user), 
                    test.filter(pl.col("user_id") == user), 
                    update_lgb_params=None
                )
            elif args.model_type == "xgb":
                _oof_df, _test_df = train_xgb(
                    args, 
                    train.filter(pl.col("user_id") == user),
                    test.filter(pl.col("user_id") == user), 
                    update_xgb_params=None
                )
        oof_df = pl.concat([oof_df, _oof_df])
        test_df = pl.concat([test_df, _test_df])
    else:
        if args.model_type == "lgb":
            oof_df, test_df = train_lgb(args, train, test, update_lgb_params=None)
        elif args.model_type == "xgb":
            oof_df, test_df = train_xgb(args, train, test, update_xgb_params=None)
            
    # get score
    best_score, threshold = get_score(oof_df[args.target_col], oof_df["pred"], step=0.005, return_threshold=True, disable=False,)
    print('\033[32m'+"====== CV score ======"+'\033[0m')
    print('\033[32m'+f'{best_score} (threshold: {threshold})'+'\033[0m')
    
    # save data
    ## oof_df
    oof_df = oof_df.sort(by="index")
    oof_df[["index", "pred"]].write_csv(args.save_dir+f"oof_df_{args.filename}.csv", has_header=True)
    
    ## test_df
    ### for ensemble
    test_df = test_df.sort("index")
    test_df[["index", "pred"]].write_csv(args.save_dir+f"{args.filename}.csv", has_header=False)
    
    ### for submit
    test_df = test_df.with_columns(
    pl.when(pl.col("pred") > threshold)
    .then(1)
    .otherwise(0)
    .alias("pred")
    )
    test_df[["index", "pred"]].write_csv(args.save_dir+f"{args.filename}_binary.csv", has_header=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)