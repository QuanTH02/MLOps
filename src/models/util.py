import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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


def train(df, selected_columns, list_file_name, is_without):
    unique_genres = set(
        genre for sublist in df["genres"].str.split() for genre in sublist
    )
    for genre in unique_genres:
        df[genre] = df["genres"].apply(lambda x: 1 if genre in x.split() else 0)
    df = df.drop(columns=["genres"])

    selected_columns = selected_columns + list(unique_genres)
    df = df[selected_columns]

    genre_columns = list(unique_genres)
    genre_data = df[genre_columns]
    scaler = StandardScaler()
    genre_data_scaled = scaler.fit_transform(genre_data)

    fa = FactorAnalyzer()
    fa.fit(genre_data_scaled)
    eigenvalues, _ = fa.get_eigenvalues()
    n_factors = sum(eigenvalues > 1)
    print(f"Number of factors to retain: {n_factors}")

    fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
    fa.fit(genre_data_scaled)
    factor_scores = fa.transform(genre_data_scaled)

    factor_scores_df = pd.DataFrame(
        factor_scores, columns=[f"Factor{i+1}" for i in range(n_factors)]
    )

    df = pd.concat([df, factor_scores_df], axis=1)

    df = df.drop(columns=genre_columns)

    mpaa_label_encoder = LabelEncoder()
    country_label_encoder = LabelEncoder()
    df["mpaa"] = mpaa_label_encoder.fit_transform(df["mpaa"])
    df["country"] = country_label_encoder.fit_transform(df["country"])
    df.to_csv(PREPROCESS_DATA_NAME_OPENING, index=False)
    X = df.drop("domestic_box_office", axis=1)
    y = df["domestic_box_office"]
    y_log = np.log(y)

    correlation_threshold = 0.2
    selected_features = [
        column
        for column in X.columns
        if abs(pearsonr(X[column], y_log)[0]) > correlation_threshold
    ]

    X = X[selected_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    numeric_features = selected_features
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ]
    )

    def grid_search(model, param_grid):
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", model),
            ]
        )
        search = GridSearchCV(
            pipeline, param_grid, cv=5, n_jobs=-1, scoring="neg_mean_squared_error"
        )
        search.fit(X_train, y_train)
        return search

    param_grid_rf = {
        "regressor__n_estimators": RF_N_ESTIMATORS,
        "regressor__max_depth": RF_MAX_DEPTH,
        "regressor__min_samples_split": RF_MIN_SAMPLES_SPLIT,
    }

    param_grid_gb = {
        "regressor__n_estimators": GB_N_ESTIMATORS,
        "regressor__max_depth": GB_MAX_DEPTH,
        "regressor__learning_rate": GB_LEARNING_RATE,
    }

    models = [
        (RandomForestRegressor(random_state=42), param_grid_rf),
        (GradientBoostingRegressor(random_state=42), param_grid_gb),
    ]

    best_score = float("inf")
    best_model = None
    best_params = None

    index_file_name = 0
    for model, param_grid in models:
        search = grid_search(model, param_grid)
        best_score = -search.best_score_
        best_model = search.best_estimator_
        best_params = search.best_params_
        with open(list_file_name[index_file_name], "wb") as f:
            pickle.dump(best_model, f)
        index_file_name += 1

        print(f"Best model: {best_model}")
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score}")

        y_pred_log = best_model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_test_actual = np.expm1(y_test)

        mse = mean_squared_error(y_test_actual, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, y_pred)
        r2 = r2_score(y_test_actual, y_pred)
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R^2 Score: {r2}")

        scores = cross_val_score(
            best_model, X, y_log, cv=5, scoring="neg_mean_squared_error"
        )
        rmse_scores = np.sqrt(-scores)
        print(f"Cross-validated RMSE scores: {rmse_scores}")
        print(f"Mean RMSE: {rmse_scores.mean()}")
        print(f"Standard deviation of RMSE: {rmse_scores.std()}")
        with open("gridsearch_result/result_without_opening.txt", "a") as f:
            print(f"Best model: {best_model}", file=f)
            print(f"Best parameters: {best_params}", file=f)
            print(f"Best score: {best_score}", file=f)
            print(f"Mean Squared Error (MSE): {mse}", file=f)
            print(f"Root Mean Squared Error (RMSE): {rmse}", file=f)
            print(f"Mean Absolute Error (MAE): {mae}", file=f)
            print(f"R^2 Score: {r2}", file=f)
            print(f"Cross-validated RMSE scores: {rmse_scores}", file=f)
            print(f"Mean RMSE: {rmse_scores.mean()}", file=f)
            print(f"Standard deviation of RMSE: {rmse_scores.std()}", file=f)
            print(
                "----------------------------------------------------------------\n\n",
                file=f,
            )

    with open(EFA + MPAA_LABEL_ENCODER, "wb") as f:
        pickle.dump(mpaa_label_encoder, f)
    with open(EFA + COUNTRY_LABEL_ENCODER, "wb") as f:
        pickle.dump(country_label_encoder, f)
    with open(EFA + SCALER, "wb") as f:
        pickle.dump(scaler, f)
    with open(EFA + FACTOR_ANALYZER, "wb") as f:
        pickle.dump(fa, f)
    with open(EFA + UNIQUE_GENRES, "wb") as f:
        pickle.dump(unique_genres, f)

    if is_without:
        with open(EFA + SELECTED_FEATURES_WITHOUT_OPENING_WEEK, "wb") as f:
            pickle.dump(selected_features, f)
    else:
        with open(EFA + SELECTED_FEATURES, "wb") as f:
            pickle.dump(selected_features, f)
