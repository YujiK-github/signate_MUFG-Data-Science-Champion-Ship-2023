import numpy as np
import polars as pl
from typing import Tuple
import lightgbm as lgb
import xgboost as xgb
from feature_engineering import apply_fe_per_fold
import warnings
warnings.simplefilter("ignore")


def train_lgb(CFG, train: pl.DataFrame, test: pl.DataFrame, update_lgb_params: dict|None) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """training and predict with lightGBM

    Args:
        CFG: configuration
        train (pl.DataFrame): training data
        test (pl.DataFrame): test data
        update_lgb_params (dict): update parameters for lightGBM

    Returns:
        tuple(pl.DataFrame, pl.DataFrame): oof_df with prediction, test_df with prediction
    """
    # parameters
    lgb_params = {
        "task":"train",
        "objective": "binary",
        "boosting":"gbdt",
        "learning_rate": 0.05,
        "num_leaves": int((2**6) * 0.7), # max number of leaves in one tree
        "max_depth": 6, # default -1, int: limit the max depth for tree model  ### xgboost, catboostに合わせる
        "min_child_weight":1e-3, # double: minimal sum hessian in one leaf
        "min_data_in_leaf":20, # minimal number of data in one leaf
        "colsample_bytree":0.4, # 0 < "colsample_bytree" < 1
        #: LightGBM will randomly select a subset of features on each iteration (tree) if feature_fraction is smaller than 1.0
        "lambda": 0, #lambda_l2 >= 0.0: L2 regularization
        "subsample":1, #0.0 < bagging_fraction <= 1.0
        "num_threads": 4,
        "metric": 'binary_logloss',
        "seed" : CFG.seed,
        "verbosity": -1, 
    }
    if update_lgb_params is not None:
        lgb_params.update(update_lgb_params)
    
    
    preds, oof_df = [], pl.DataFrame()
    for fold in range(CFG.n_splits):
        X_train, X_valid, test_df = apply_fe_per_fold(CFG, train, test, fold)
        categorical_features = [col+"_category" for col in CFG.categorical_features]
        lgb_train = lgb.Dataset(
            data=X_train[CFG.use_features], label=X_train[CFG.target_col],
        )
        lgb_valid = lgb.Dataset(
            data=X_valid[CFG.use_features], label=X_valid[CFG.target_col],
        )
        # train
        model = lgb.train(
            params = lgb_params,
            train_set = lgb_train,
            num_boost_round = 10000,
            valid_sets = [lgb_valid],
            callbacks=[
                lgb.early_stopping(verbose=CFG.show_log, stopping_rounds=100),
                lgb.log_evaluation(period=200 if CFG.show_log else -1, show_stdv=False)
            ]
        )
        
        # valid
        X_valid = X_valid.with_columns(
            [
                pl.Series(model.predict(data=X_valid[CFG.use_features], num_iteration=model.best_iteration)).alias("pred"),
            ]
        )
            
        # oof_df
        oof_df = pl.concat([oof_df, X_valid])
        
        # predict
        preds.append(model.predict(data=test_df[CFG.use_features], num_iteration=model.best_iteration))
        
    test_df = test_df.with_columns(
        [
            pl.Series(np.mean(preds, axis=0)).alias("pred")
        ]
    )
    return oof_df, test_df



def train_xgb(CFG, train: pl.DataFrame, test: pl.DataFrame, update_xgb_params: dict) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """training and predict with XGBoost

    Args:
        CFG: configuration
        train (pl.DataFrame): training data
        test (pl.DataFrame): test data
        update_xgb_params (dict): update parameters for XGBoost

    Returns:
        tuple(pl.DataFrame, pl.DataFrame): oof_df with prediction, test_df with prediction
    """
    # parameters
    xgb_params = {
        "booster": "gbtree",
        "verbosity": 1,
        "nthread": 4,
        "eta": 0.3,
        "max_depth": 6,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": CFG.seed,
    }
    if update_xgb_params is not None:
        xgb_params.update(update_xgb_params)
    
    
    preds, oof_df = [], pl.DataFrame()
    for fold in range(CFG.n_splits):
        X_train, X_valid, test_df = apply_fe_per_fold(CFG, train, test, fold)
        d_train = xgb.DMatrix(
            data=X_train[CFG.use_features], label=X_train[CFG.target_col], nthread=-1, enable_categorical=True
        )
        d_valid = xgb.DMatrix(
            data=X_valid[CFG.use_features], label=X_valid[CFG.target_col], nthread=-1, enable_categorical=True
        )
        # train
        model = xgb.train(
            params = xgb_params,
            dtrain = d_train,
            num_boost_round = 10000,
            evals = [(d_valid, 'valid'), (d_train, 'train')],
            early_stopping_rounds = 100,
        )
        
        # valid
        X_valid = X_valid.with_columns(
            [
                pl.Series(model.predict(data=xgb.DMatrix(data=X_valid[CFG.use_features], enable_categorical=True),ntree_limit=model.best_iteration)).alias("pred")
            ]
        )
        oof_df = pl.concat([oof_df, X_valid])
        
        # test
        preds.append(
            model.predict(data=xgb.DMatrix(data=test_df[CFG.use_features], enable_categorical=True), ntree_limit=model.best_iteration)
        )
        
    test_df = test_df.with_columns(
        [
            pl.Series(np.mean(preds, axis=0)).alias("pred")
        ]
    )
    return oof_df, test_df