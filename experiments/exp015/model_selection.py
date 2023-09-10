import pandas as pd
from tqdm.auto import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def kfold(CFG, df: pd.DataFrame) -> pd.DataFrame:
    """kfold

    Args:
        CFG: configuration
        df (pd.DataFrame): dataframe to apply kfold

    Returns:
        pd.DataFrame: dataframe applied kfold
    """
    _df = pd.DataFrame()
    for user in tqdm(df["user_id"].unique(), desc="Making fold"):
        _df = pd.concat([_df, kfold_per_user(CFG, df, user)])
    return _df.reset_index(drop=True)


def kfold_per_user(CFG, df: pd.DataFrame, user: str) -> pd.DataFrame:
    """kfold per user

    Args:
        CFG: configuration
        df (pd.DataFrame): dataframe to apply kfold per user
        user (str): kfold against this user's data.

    Returns:
        pd.DataFrame: dataframe applied kfold per user
    """
    tmp = df[df["user_id"] == user].reset_index(drop=True)
    tmp = tmp.sort_values("index")
    tmp["id"] = range(len(tmp))
    skf = MultilabelStratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
    for i, (_, val) in enumerate(skf.split(X=tmp, y=tmp[["is_fraud?", "card_id"]])):
        tmp.loc[val, "fold"] = int(i)
    return tmp.drop("id", axis=1)