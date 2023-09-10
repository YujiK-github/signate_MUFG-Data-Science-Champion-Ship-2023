import polars as pl
from typing import Tuple
from preprocessing import CustomOrdinalEncoder

def apply_fe(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [   
            # str -> float
            pl.col("amount").apply(lambda x: x[1:]).cast(pl.Float64),
            pl.col("total_debt").apply(lambda x: x[1:]).cast(pl.Float64),
            pl.col("credit_limit").apply(lambda x: x[1:]).cast(pl.Float64),
            pl.col("yearly_income_person").apply(lambda x: x[1:]).cast(pl.Float64),
            pl.col("per_capita_income_zipcode").apply(lambda x: x[1:]).cast(pl.Float64),
            
            # str -> Datetime
            pl.col("expires").str.strptime(dtype=pl.Date, format="%m/%Y"),
            pl.col("acct_open_date").str.strptime(dtype=pl.Date, format="%m/%Y"),
            
            # bool
            (pl.col("zip") == pl.col("zipcode")).alias("same_zipcode_as_zip"),
            pl.when((pl.col("merchant_city").is_null())&(pl.col("merchant_city") != "ONLINE")) ## TODO: 上手くまとめられないかな
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("city_is_not_America"),            
        ]
    )
    
    df = df.with_columns(
        [
            # Datetime -> Month, Year
            pl.col("expires").dt.year().suffix("_year"),
            pl.col("expires").dt.month().suffix("_month"),
            pl.col("acct_open_date").dt.year().suffix("_year"),
            pl.col("acct_open_date").dt.month().suffix("_month"),        
        ]
    )
    return df


def apply_fe_per_fold(CFG, train: pl.DataFrame, test: pl.DataFrame, fold: int) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    # data split
    X_train = train.filter(
        pl.col("fold") != fold
    )
    X_valid = train.filter(
        pl.col("fold") == fold
    )
    test_df = test.clone()

    
    # feature_engineering
    ## Average transaction amount when there is fraudulent use for each card_id
    mean_map = X_train.groupby(by=["user_id", "card_id", "is_fraud?"]).agg(
        pl.col("amount").mean()
    ).filter(
        pl.col("is_fraud?") == 0
    ).rename(
        {
            "amount": "NonFraudAvgAmount_per_user_card"
        }
    )[["user_id", "card_id", "NonFraudAvgAmount_per_user_card"]]
    
    X_train = X_train.join(mean_map, on=["user_id", "card_id"], how="left")
    X_valid = X_valid.join(mean_map, on=["user_id", "card_id"], how="left")
    test_df = test_df.join(mean_map, on=["user_id", "card_id"], how="left")
    
    
    # count_encoding
    count_map = X_train.groupby(by="merchant_id").count().rename({"count":"merchant_id_count_encoding"})
    X_train = X_train.join(count_map, on="merchant_id", how="left")
    X_valid = X_valid.join(count_map, on="merchant_id", how="left")
    test_df = test_df.join(count_map, on="merchant_id", how="left")
    
    
    if CFG.model_type == "xgb":
        # OrdinalEncoder
        ## https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html
        oe = CustomOrdinalEncoder(encoded_missing_value=-1)
        X_train = pl.concat([X_train, 
                            oe.fit_transform(X_train[CFG.categorical_features])
                            ], how="horizontal")
        X_valid = pl.concat([X_valid, 
                            oe.transform(X_valid[CFG.categorical_features])
                            ], how="horizontal")
        test_df = pl.concat([test_df, 
                            oe.transform(test_df[CFG.categorical_features])
                            ], how="horizontal")
        
    elif CFG.model_type == "lgb":
        # cast Categorical
        ## https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
        for col in CFG.categorical_features:
            X_train = X_train.with_columns(
                [
                    pl.col(col).cast(pl.Utf8).cast(pl.Categorical).suffix("_category")
                ]
            )
            X_valid = X_valid.with_columns(
                [
                    pl.col(col).cast(pl.Utf8).cast(pl.Categorical).suffix("_category")
                ]
            )
            test_df = test_df.with_columns(
                [
                    pl.col(col).cast(pl.Utf8).cast(pl.Categorical).suffix("_category")
                ]
            )

    return X_train, X_valid, test_df