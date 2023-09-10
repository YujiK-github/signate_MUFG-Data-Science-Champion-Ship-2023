import numpy as np
import pandas as pd
from typing import Tuple
import lightgbm as lgb
import xgboost as xgb
from feature_engineering import apply_fe_per_fold
import warnings
warnings.simplefilter("ignore")

def train_lgb(CFG, train: pd.DataFrame, test: pd.DataFrame, update_lgb_params: dict|None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """training and predict with lightGBM

    Args:
        CFG: configuration
        train (pd.DataFrame): training data
        test (pd.DataFrame): test data
        update_lgb_params (dict): update parameters for lightGBM

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): oof_df with prediction, test_df with prediction
    """
    # parameters
    lgb_params = {
        "task":"train",
        "objective": "binary",
        "boosting":"gbdt",
        "num_iterations": 10000,
        "learning_rate": 0.05,
        "num_leaves": int((2**3) * 0.7),
        "max_depth": 3,
        "min_child_weight":1e-3,
        "min_data_in_leaf":20,
        "colsample_bytree":0.4,
        "lambda": 0,
        "subsample":1,
        "num_threads": 4,
        "metric": 'binary_logloss',
        "seed" : CFG.seed,
        "verbosity": -1, 
    }
    if update_lgb_params is not None:
        lgb_params.update(update_lgb_params)
        
    preds, oof_df = [], pd.DataFrame()
    for fold in range(CFG.n_splits):
        # data
        X_train, X_valid, test_df = apply_fe_per_fold(CFG, train, test, fold)
        categorical_features = [col for col in CFG.use_features if "_category" in col]
        lgb_train = lgb.Dataset(X_train[CFG.use_features], X_train[CFG.target_col], categorical_feature = categorical_features,)
        lgb_valid = lgb.Dataset(X_valid[CFG.use_features], X_valid[CFG.target_col], categorical_feature = categorical_features,)
        
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
        X_valid["pred"] = model.predict(X_valid[CFG.use_features], num_iteration=model.best_iteration)
        
        # oof
        oof_df = pd.concat([oof_df, X_valid])
        
        # predict
        preds.append(model.predict(test_df[CFG.use_features], num_iteration=model.best_iteration))
        
    test_df["pred"] = np.mean(preds, axis=0)
    return oof_df, test_df



def train_xgb(CFG, train: pd.DataFrame, test: pd.DataFrame, update_xgb_params: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        "verbosity": 0,
        "nthread": 4,
        "eta": 0.3,
        "max_depth": 6,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": CFG.seed,
    }
    if update_xgb_params is not None:
        xgb_params.update(update_xgb_params)
    
    
    preds, oof_df = [], pd.DataFrame()
    for fold in range(CFG.n_splits):
        # data
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
            evals = [(d_train, 'train'), (d_valid, 'valid')],
            early_stopping_rounds = 100,
            verbose_eval=False,
        )
        
        # valid
        X_valid["pred"] = model.predict(data=xgb.DMatrix(data=X_valid[CFG.use_features], enable_categorical=True))
        
        # oof
        oof_df = pd.concat([oof_df, X_valid])
        
        # predict
        preds.append(model.predict(xgb.DMatrix(data=test_df[CFG.use_features], enable_categorical=True)))
        
    test_df["pred"] = np.mean(preds, axis=0)
    return oof_df, test_df