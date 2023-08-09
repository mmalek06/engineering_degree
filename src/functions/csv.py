import pandas as pd


def stratified_split(df: pd.DataFrame, category_name: str, test_size=0.2):
    groups = df.groupby(category_name)
    train_dfs = []
    valid_dfs = []

    for _, group in groups:
        valid_size = int(len(group) * test_size)
        valid_size = max(valid_size, 1)
        valid_df = group.sample(n=valid_size)
        train_df = group.drop(valid_df.index)

        train_dfs.append(train_df)
        valid_dfs.append(valid_df)

    train_df = pd.concat(train_dfs)
    valid_df = pd.concat(valid_dfs)

    return train_df, valid_df
