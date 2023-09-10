import polars as pl

def load_data(CFG) -> pl.DataFrame:
    """load data

    Args:
        CFG : configuration

    Returns:
        pl.DataFrame: dataframe that train and test concat
    """
    train = pl.read_csv(CFG.data_dir+"train.csv")
    test = pl.read_csv(CFG.data_dir+"test.csv")
    card = pl.read_csv(CFG.data_dir+"card.csv")
    user = pl.read_csv(CFG.data_dir+"user.csv")
    
    
    if CFG.debug:
        train = train.sample(n=10000, seed=CFG.seed)
        test = test.sample(n=1000, seed=CFG.seed)

    # prepare concat
    train = train.with_columns(
        pl.lit("train").alias("flag")
    )
    test = test.with_columns(
        [
            pl.lit(None, dtype=pl.Int64).alias("is_fraud?"),
            pl.lit("test").alias("flag"),
        ]
    )

    # concat
    all_data = pl.concat([train, test], how="align")
    
    # merge
    all_data = all_data.join(
        card, on=["user_id", "card_id"], how="left"
    )
    all_data = all_data.join(
        user, on="user_id", how="left"
    )
    
    return all_data