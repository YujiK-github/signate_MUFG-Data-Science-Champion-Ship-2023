import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import argparse
import sys
sys.path.append("G:/マイドライブ/signate_MUFJ2023/experiments/exp015/")
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
    parser.add_argument("--n_splits", default=15, type=int, required=False)
    parser.add_argument("--filename", default="expXXX", type=str, required=False)
    parser.add_argument("--save_dir", default="G:/マイドライブ/signate_MUFJ2023/exp/", type=str, required=False)
    parser.add_argument("--data_dir", default="G:/マイドライブ/signate_MUFJ2023/data/", type=str, required=False)
    parser.add_argument("--debug", action="store_true", required=False)
    parser.add_argument("--target_col", default="is_fraud?", type=str, required=False)
    parser.add_argument("--training_per_user_id", action="store_false", required=False)
    parser.add_argument("--show_log", action="store_true", required=False)
    parser.add_argument("--model_type", default="lgb", type=str, required=False, choices=["lgb", "xgb"])
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
    train = all_data[all_data["flag"] == "train"]
    test = all_data[all_data["flag"] == "test"]
    
    # apply kfold
    train = kfold(args, train)
    
    # train
    args.use_features = args.numerical_features + [col+"_category" for col in args.categorical_features]
    
    if args.training_per_user_id:
        oof_df, test_df = pd.DataFrame(), pd.DataFrame()
        for user in tqdm(train["user_id"].unique()):
            if args.model_type == "lgb":
                _oof_df, _test_df = train_lgb(
                    args, 
                    train[train["user_id"] == user], 
                    test[test["user_id"] == user], 
                    update_lgb_params=None
                )
            elif args.model_type == "xgb":
                _oof_df, _test_df = train_xgb(
                    args, 
                    train[train["user_id"] == user],
                    test[test["user_id"] == user], 
                    update_xgb_params=None
                )
            oof_df = pd.concat([oof_df, _oof_df])
            test_df = pd.concat([test_df, _test_df])
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
    oof_df = oof_df.sort_values(by="index", ignore_index=True)
    oof_df[["index", "pred"]].to_csv(args.save_dir+f"oof_df_{args.filename}_{args.model_type}.csv", index=False)
    
    ## test_df
    ### for ensemble
    test_df = test_df.sort_values(by="index", ignore_index=True)
    test_df[["index", "pred"]].to_csv(args.save_dir+f"{args.filename}_{args.model_type}.csv", index=False)
    
    ### for submit
    test_df["pred"] = np.where(test_df["pred"] > threshold, 1, 0)
    test_df[["index", "pred"]].to_csv(args.save_dir+f"{args.filename}_binary_{args.model_type}.csv", index=False, header=None)


if __name__ == "__main__":
    args = parse_args()
    main(args)