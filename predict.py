import os

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from utils.constants import SEASON
from utils.get_ids import get_player_name

actual_points = None


def get_features(gameweek: int) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("final_df.csv")
    df = df[df["gw"] == gameweek].copy()

    ids = df["player_id"]
    global actual_points
    actual_points = df["total_points"]

    features = df.drop(
        columns=["player_id", "team_id", "opponent_team_id", "total_points"]
    )

    # Standardize the features
    # but not the gameweek column
    gws = features["gw"]
    features = features.drop(columns=["gw"])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features = pd.concat(
        [
            pd.DataFrame(features_scaled, columns=features.columns),
            gws.reset_index(drop=True),
        ],
        axis=1,
    )

    return features, ids


def predict(gameweek: int) -> pd.DataFrame:
    model = tf.keras.models.load_model("best_model.keras")
    features, ids = get_features(gameweek)
    predictions = model.predict(features)

    predictions_df = pd.DataFrame(predictions, columns=["predicted_points"])

    result = pd.concat(
        [
            ids.reset_index(drop=True),
            features[["gw"]].reset_index(drop=True),
            predictions_df,
        ],
        axis=1,
    )

    result["web_name"] = result["player_id"].apply(get_player_name, args=(SEASON,))
    return result


def main():
    gameweek = 38
    predictions = predict(gameweek)

    global actual_points
    actual_points = actual_points.tolist()
    predictions["actual_points"] = actual_points
    predictions["difference"] = (
        predictions["predicted_points"] - predictions["actual_points"]
    )

    os.makedirs("predictions", exist_ok=True)
    print(f"Saving predictions for gameweek {gameweek}...")

    predictions.to_csv(f"predictions/predictions_gw_{gameweek}.csv", index=False)


if __name__ == "__main__":
    main()
