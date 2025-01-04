DATA_FILE_NAME = "../../data/processed/final_merged.csv"
PREPROCESS_DATA_NAME = "../../data/processed/preprocess_data.csv"
PREPROCESS_DATA_NAME_OPENING = (
    "../../data/processed/preprocess_data_without_opening_week.csv"
)

EFA = "model_efa/"
MODEL_EFR = "../models/model_efa/"

MPAA_LABEL_ENCODER = "mpaa_label_encoder.pkl"
COUNTRY_LABEL_ENCODER = "country_label_encoder.pkl"
SCALER = "scaler.pkl"
FACTOR_ANALYZER = "factor_analyzer.pkl"
UNIQUE_GENRES = "unique_genres.pkl"
SELECTED_FEATURES = "selected_features.pkl"
SELECTED_FEATURES_WITHOUT_OPENING_WEEK = "selected_features_without_opening_week.pkl"

MODEL_RF = "model_rf.pkl"
MODEL_GB = "model_gb.pkl"

MODEL_RF_WITHOUT_OPENING_WEEK = "model_rf_without_opening_week.pkl"
MODEL_GB_WITHOUT_OPENING_WEEK = "model_gb_without_opening_week.pkl"

RF_N_ESTIMATORS = [50, 100, 150]
RF_MAX_DEPTH = [None, 10, 20, 30]
RF_MIN_SAMPLES_SPLIT = [2, 5, 10]

GB_N_ESTIMATORS = [50, 100, 150]
GB_MAX_DEPTH = [3, 5, 7]
GB_LEARNING_RATE = [0.01, 0.1, 0.2]
