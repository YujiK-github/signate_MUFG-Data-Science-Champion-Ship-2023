import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import OrdinalEncoder

def apply_fe(df: pd.DataFrame) -> pd.DataFrame:
    """apply feature engineering

    Args:
        df (pd.DataFrame): dataframe to apply feature engineering

    Returns:
        pd.DataFrame: dataframe applied feature engineering
    """
    # str -> float
    for col in ["amount", "total_debt", "credit_limit", "yearly_income_person", "per_capita_income_zipcode"]:
        df[col] = df[col].apply(lambda x: x[1:]).astype(float)
        
    # str -> datetime
    for col in ["expires", "acct_open_date"]:
        df[col] = pd.to_datetime(df[col], format="%m/%Y")
        df[col+"_year"] = df[col].dt.year
        df[col+"_month"] = df[col].dt.month

            
    # user_id + card_id
    df["user_card_id"] = df["user_id"].astype(str) + "-" + df["card_id"].astype(str)
    
    # bool
    df["same_zipcode_as_zip"] = (df["zip"] == df["zipcode"])
    df["city_is_not_America"] = ((df["zip"].isnull())&(df["merchant_city"] != "ONLINE"))
    return df


def apply_fe_per_fold(CFG, train: pd.DataFrame, test: pd.DataFrame, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """apply feature engineering per fold

    Args:
        CFG: configuration
        train (pd.DataFrame): training dataframe to apply feature engineering per fold
        test (pd.DataFrame): test dataframe to apply feature engineering per fold
        fold (int): fold

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: dataframes applied feature engineering per fold
    """
    # data split
    X_train = train[train["fold"] != fold].reset_index(drop=True)
    X_valid = train[train["fold"] == fold].reset_index(drop=True)
    test_df = test.copy()
    
    # Average transaction amount when there is fraudulent use for each card_id
    tmp = X_train.groupby(by=["user_card_id", "is_fraud?"])["amount"].mean().reset_index()
    tmp_1 = tmp[tmp["is_fraud?"] == 0].rename(columns={"amount":"NonFraudAvgAmount_per_user_card"})[["user_card_id", "NonFraudAvgAmount_per_user_card"]]
    X_train = X_train.merge(tmp_1, on="user_card_id", how="left")
    X_valid = X_valid.merge(tmp_1, on="user_card_id", how="left")
    test_df = test_df.merge(tmp_1, on="user_card_id", how="left")
        
    # count_encoding
    for col in ["merchant_id"]:
        count_map = X_train[col].value_counts().to_dict()
        X_train[col+"_count_encoding"] = X_train[col].map(count_map)
        X_valid[col+"_count_encoding"] = X_valid[col].map(count_map)
        test_df[col+"_count_encoding"] =test_df[col].map(count_map)

    if CFG.model_type == "lgb":
        # OrdinalEncoder
        ## https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html
        oe = OrdinalEncoder(categories="auto",
                            handle_unknown="use_encoded_value",
                            unknown_value=99999,
                            encoded_missing_value=np.nan
                            )
        CFG.categorical_features_ = [feature + "_category" for feature in CFG.categorical_features]
        X_train[CFG.categorical_features_] = oe.fit_transform(X_train[CFG.categorical_features].values)
        X_valid[CFG.categorical_features_] = oe.transform(X_valid[CFG.categorical_features].values)
        test_df[CFG.categorical_features_] = oe.transform(test_df[CFG.categorical_features].values)
        
        
    elif CFG.model_type == "xgb":
        # cast Categorical
        ## https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
        for col in CFG.categorical_features:
            X_train[col+"_category"] = X_train[col].astype("category")
            X_valid[col+"_category"] = X_valid[col].astype("category")
            test_df[col+"_category"] = test_df[col].astype("category")

    return X_train, X_valid, test_df