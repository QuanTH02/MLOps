import pandas as pd


from constant import (
    DATA_FILE_NAME,
    EFA,
    MODEL_GB_WITHOUT_OPENING_WEEK,
    MODEL_RF_WITHOUT_OPENING_WEEK,
    MODEL_RF,
    MODEL_GB,
)
from util import train

if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE_NAME)
    selected_columns = [
        "month",
        "year",
        "mpaa",
        "budget",
        "runtime",
        "screens",
        "opening_week",
        "domestic_box_office",
        "user_vote",
        "ratings",
        "critic_vote",
        "meta_score",
        "country",
        "sequel",
    ]
    list_file_name = [EFA + MODEL_RF, EFA + MODEL_GB]

    selected_columns_without = [
        "month",
        "year",
        "mpaa",
        "budget",
        "runtime",
        "screens",
        "domestic_box_office",
        "critic_vote",
        "meta_score",
        "country",
        "sequel",
    ]
    list_file_name_without = [
        EFA + MODEL_RF_WITHOUT_OPENING_WEEK,
        EFA + MODEL_GB_WITHOUT_OPENING_WEEK,
    ]

    train(df.copy(), selected_columns, list_file_name, False)
    train(df.copy(), selected_columns_without, list_file_name_without, True)
