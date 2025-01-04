import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
from scipy.stats import pearsonr
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from factor_analyzer import FactorAnalyzer
import pickle

from constant import (
    DATA_FILE_NAME,
    PREPROCESS_DATA_NAME,
    PREPROCESS_DATA_NAME_OPENING,
    EFA,
    MPAA_LABEL_ENCODER,
    COUNTRY_LABEL_ENCODER,
    SCALER,
    FACTOR_ANALYZER,
    UNIQUE_GENRES,
    SELECTED_FEATURES,
    MODEL_GB_WITHOUT_OPENING_WEEK,
    MODEL_RF_WITHOUT_OPENING_WEEK,
    MODEL_RF,
    MODEL_GB,
    SELECTED_FEATURES_WITHOUT_OPENING_WEEK,
    RF_N_ESTIMATORS,
    RF_MIN_SAMPLES_SPLIT,
    RF_MAX_DEPTH,
    GB_MAX_DEPTH,
    GB_LEARNING_RATE,
    GB_N_ESTIMATORS,
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
