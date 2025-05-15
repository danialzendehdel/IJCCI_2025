import numpy as np
import pandas as pd
import os

class PNetDataLoader:
    def __init__(self, path):

        if not os.path.exists(path):
            assert False,f"File {path} does not exist"

        df = pd.read_csv(path)

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

        df.dropna(inplace=True)
        df = df.reset_index()
        df = df.sort_values("datetime").drop_duplicates("datetime")
        df["solar"] = df["solar"].clip(lower=0)
        df["P"] = ((df["load"] - df["solar"]) / (8.5) )* 18 / 20  # x kw/ 8.5 kw * 18 #⚠️TODO: CHECK THIS UNIT

        self.df = df
        self.current_index = 0
        print(f"max: {df["P"].max()}")
        print(f"min: {df["P"].min()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
            return self.df["P"].iloc[index]
