import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=FutureWarning)

from constant import (
    MODEL_EFR,
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
)


def predict_with_feature_selection(
    model_file_name,
    month,
    year,
    mpaa,
    budget,
    runtime,
    screens,
    opening_week,
    user_vote,
    ratings,
    critic_vote,
    meta_score,
    sequel,
    genres,
    country,
):
    movie = {}

    movie["month"] = float(month)
    movie["year"] = float(year)
    movie["mpaa"] = mpaa
    movie["budget"] = float(budget)
    movie["runtime"] = float(runtime)
    movie["screens"] = float(screens)
    movie["opening_week"] = float(opening_week)
    movie["user_vote"] = float(user_vote)
    movie["ratings"] = float(ratings)
    movie["critic_vote"] = float(critic_vote)
    movie["meta_score"] = float(meta_score)
    movie["sequel"] = float(sequel)
    movie["genres"] = genres
    movie["country"] = country

    with open(model_file_name, "rb") as f:
        model = pickle.load(f)
    with open(MODEL_EFR + MPAA_LABEL_ENCODER, "rb") as f:
        mpaa_label_encoder = pickle.load(f)
    with open(MODEL_EFR + COUNTRY_LABEL_ENCODER, "rb") as f:
        country_label_encoder = pickle.load(f)
    with open(MODEL_EFR + SCALER, "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_EFR + FACTOR_ANALYZER, "rb") as f:
        fa = pickle.load(f)
    with open(MODEL_EFR + UNIQUE_GENRES, "rb") as f:
        unique_genres = pickle.load(f)
    with open(MODEL_EFR + SELECTED_FEATURES, "rb") as f:
        selected_features = pickle.load(f)

    movie["mpaa"] = mpaa_label_encoder.transform([movie["mpaa"]])[0]
    movie["country"] = country_label_encoder.transform([movie["country"]])[0]

    new_movie_genres = np.array(
        [
            1 if genre in movie.get("genres", "").split() else 0
            for genre in unique_genres
        ]
    ).reshape(1, -1)
    new_movie_genres_scaled = scaler.transform(new_movie_genres)
    new_movie_factors = fa.transform(new_movie_genres_scaled)

    movie.update(
        {
            f"Factor{i+1}": new_movie_factors[0, i]
            for i in range(new_movie_factors.shape[1])
        }
    )

    movie_df = pd.DataFrame([movie])
    movie_df = movie_df[selected_features]
    prediction_log = model.predict(movie_df)
    prediction = np.expm1(prediction_log)

    if prediction[0]:
        return True


def predict_with_feature_selection_without_opening_week(
    model_file_name,
    month,
    year,
    mpaa,
    budget,
    runtime,
    screens,
    critic_vote,
    meta_score,
    sequel,
    genres,
    country,
):
    movie = {}

    movie["month"] = float(month)
    movie["year"] = float(year)
    movie["mpaa"] = mpaa
    movie["budget"] = float(budget)
    movie["runtime"] = float(runtime)
    movie["screens"] = float(screens)
    movie["critic_vote"] = float(critic_vote)
    movie["meta_score"] = float(meta_score)
    movie["sequel"] = float(sequel)
    movie["genres"] = genres
    movie["country"] = country

    with open(model_file_name, "rb") as f:
        model = pickle.load(f)
    with open(MODEL_EFR + MPAA_LABEL_ENCODER, "rb") as f:
        mpaa_label_encoder = pickle.load(f)
    with open(MODEL_EFR + COUNTRY_LABEL_ENCODER, "rb") as f:
        country_label_encoder = pickle.load(f)
    with open(MODEL_EFR + SCALER, "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_EFR + FACTOR_ANALYZER, "rb") as f:
        fa = pickle.load(f)
    with open(MODEL_EFR + UNIQUE_GENRES, "rb") as f:
        unique_genres = pickle.load(f)
    with open(MODEL_EFR + SELECTED_FEATURES_WITHOUT_OPENING_WEEK, "rb") as f:
        selected_features = pickle.load(f)

    movie["mpaa"] = mpaa_label_encoder.transform([movie["mpaa"]])[0]
    movie["country"] = country_label_encoder.transform([movie["country"]])[0]

    new_movie_genres = np.array(
        [
            1 if genre in movie.get("genres", "").split() else 0
            for genre in unique_genres
        ]
    ).reshape(1, -1)
    new_movie_genres_scaled = scaler.transform(new_movie_genres)
    new_movie_factors = fa.transform(new_movie_genres_scaled)

    movie.update(
        {
            f"Factor{i+1}": new_movie_factors[0, i]
            for i in range(new_movie_factors.shape[1])
        }
    )

    movie_df = pd.DataFrame([movie])
    movie_df = movie_df[selected_features]
    prediction_log = model.predict(movie_df)
    prediction = np.expm1(prediction_log)

    if prediction[0]:
        return True
