import os


def add_path_to_data(df, data_path):
    df["Path"] = df["Id"].apply(
        lambda id: os.path.join(data_path, f"{id}.jpg")
    )
    return df
