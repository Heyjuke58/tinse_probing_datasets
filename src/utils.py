import pandas as pd
import time

def set_new_index(df: pd.DataFrame) -> pd.DataFrame:
    df["idx"] = pd.Int64Index(range(df.shape[0]))
    return df.set_index("idx")


def get_timestamp() -> str:
    return time.strftime("%Y_%m_%d-%H-%M-%S")