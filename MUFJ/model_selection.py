import pandas as pd
from tqdm.auto import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def kfold(CFG, df: pd.DataFrame, check_bias: bool = True) -> pd.DataFrame:
    """kfold

    Args:
        CFG (_type_): config
        df (pd.DataFrame): kfoldを行うdataframe
        check_bias (bool, optional): foldに分けたとき、それぞれのデータが適切に割り振られているかをチェックする. Defaults to True.

    Returns:
        pd.DataFrame: kfoldを行ったdataframe
    """
    _df = pd.DataFrame()
    for user in tqdm(df["user_id"].unique(), desc="Making fold"):
        _df = pd.concat([_df, kfold_per_user(CFG, df, user)])
    if check_bias:
        check_fold_bias(CFG, _df)
    return _df.reset_index(drop=True)


def kfold_per_user(CFG, df: pd.DataFrame, user: str) -> pd.DataFrame:
    """kfold per user

    Args:
        CFG (_type_): config
        df (pd.DataFrame): _kfoldを行うデータ
        user (str): このユーザーのデータに対してkfoldを行う

    Returns:
        pd.DataFrame: kfoldを行ったデータ
    """
    tmp = df[df["user_id"] == user].reset_index(drop=True)
    tmp = tmp.sort_values("index")
    tmp["id"] = range(len(tmp))
    skf = MultilabelStratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
    for i, (_, val) in enumerate(skf.split(X=tmp, y=tmp[["is_fraud?", "card_id"]])):
        tmp.loc[val, "fold"] = int(i)
    return tmp.drop("id", axis=1)


def check_fold_bias(CFG, df: pd.DataFrame) -> None:
    """foldごとに割り振ったデータの数が偏っていないか確認する

    Args:
        CFG (_type_): config
        df (pd.DataFrame)
    """
    for user in tqdm(df["user_id"].unique(), desc="Check_fold_bias"):
        if len(df[df["user_id"] == user]["fold"].value_counts().unique()) > CFG.patience:
            print(f"Fold of User_id {user} is {df[df['user_id'] == user]['fold'].value_counts()}")
    else:
        print("The folds for each user have been correctly allocated.")