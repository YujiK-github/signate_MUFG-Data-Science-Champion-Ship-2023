import pandas as pd

def load_data(CFG) -> pd.DataFrame:
    """load data

    Args:
        CFG : configuration

    Returns:
        pd.DataFrame: dataframe that train and test concat
    """
    train = pd.read_csv(CFG.data_dir+"train.csv")
    test = pd.read_csv(CFG.data_dir+"test.csv")
    card = pd.read_csv(CFG.data_dir+"card.csv")
    user = pd.read_csv(CFG.data_dir+"user.csv")
    
    
    if CFG.debug:
        train = train.sample(n=10000, random_state=CFG.seed, ignore_index=True)
        test = test.sample(n=1000, random_state=CFG.seed, ignore_index=True)

    # prepare concat
    train["flag"] = "train"
    test["flag"] = "test"

    # concat
    all_data = pd.concat([train, test])
    
    # merge
    all_data = pd.merge(all_data, card, on=["user_id", "card_id"], how="left")
    all_data = pd.merge(all_data, user, on=["user_id"], how="left")
    
    return all_data