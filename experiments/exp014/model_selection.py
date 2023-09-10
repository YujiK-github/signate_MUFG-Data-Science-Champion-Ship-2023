import polars as pl
from tqdm.auto import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def kfold(CFG, df: pl.DataFrame) -> pl.DataFrame:
    """kfold

    Args:
        CFG: configuration
        df (pl.DataFrame): dataframe to kfold

    Returns:
        pl.DataFrame: kfolded dataframe
    """
    _df = pl.DataFrame()
    for user in tqdm(df["user_id"].unique()):
        _df = pl.concat([_df, kfold_per_user(CFG, df, user)])
    return _df.drop(columns="id")


def kfold_per_user(CFG, df: pl.DataFrame, user: str) -> pl.DataFrame:
    """kfold per user

    Args:
        CFG: configuration
        df (pl.DataFrame): dataframe to kfold per user
        user (str): kfold against this user's data.

    Returns:
        pl.DataFrame: kfolded dataframe
    """
    df_per_user = df.filter(pl.col("user_id") == user)
    df_per_user = df_per_user.with_columns(
        [
            pl.Series(range(len(df_per_user))).alias("id"),
            pl.lit(None).alias("fold")
        ]
    )
    mlskf = MultilabelStratifiedKFold(n_splits=CFG.n_splits, random_state=CFG.seed, shuffle=True)
    for i, (_, val) in enumerate(mlskf.split(X=df_per_user, y=df_per_user[["card_id", "is_fraud?"]])):
        df_per_user = df_per_user.with_columns(
            pl.when(pl.col("id").is_in(val))
            .then(pl.lit(i))
            .otherwise(pl.col("fold"))
            .alias("fold")
        )
    return df_per_user