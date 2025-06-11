import os

import pandas as pd
import tensorflow as tf
from pulp import LpMaximize, LpProblem, LpStatus, LpVariable, lpSum
from sklearn.preprocessing import StandardScaler

from utils.constants import SEASON
from utils.general import get_data_path
from utils.get_ids import get_current_player_prices, get_player_name


def get_features(gameweek: int) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv("final_df.csv")
    df = df[df["gw"] == gameweek].copy()

    ids = df["player_id"]

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
    features.to_csv("temp.csv", index=False)
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


def get_best_team(gameweek: int, team_budget: float) -> dict[list[int], float]:
    """
    Get the best team based on the predictions for a specific gameweek.
    Args:
        gameweek (int): The gameweek for which to get the best team.
    Returns:
        dict: A dictionary with player IDs as keys and their predicted points as values.
    """
    predictions = predict(gameweek)

    # Add player positions and prices to the predictions DataFrame
    player_positions = pd.read_csv(get_data_path(SEASON, "players_season_data.csv"))
    player_positions = player_positions[["player_id", "element_type"]]
    player_positions.rename(columns={"element_type": "position"}, inplace=True)

    predictions = predictions.merge(
        player_positions[["player_id", "position"]],
        on="player_id",
        how="left",
    )

    # Get current player prices
    player_prices = get_current_player_prices()
    predictions = predictions.merge(
        player_prices[["player_id", "current_cost"]],
        on="player_id",
        how="left",
    )

    position_constraints = {
        1: 2,  # Goalkeepers
        2: 5,  # Defenders
        3: 5,  # Midfielders
        4: 3,  # Forwards
    }

    # Create the linear programming problem (knapsack problem)
    model = LpProblem(name="fpl_best_team", sense=LpMaximize)
    player_vars = LpVariable.dict(
        "player",
        predictions["player_id"],
        cat="Binary",
    )

    # Objective function: maximize the sum of predicted points
    model += (
        lpSum(
            player_vars[player_id]
            * predictions.loc[
                predictions["player_id"] == player_id, "predicted_points"
            ].values[0]
            for player_id in predictions["player_id"]
        ),
        "Total_Predicted_Points",
    )

    # Constraints: total cost of players must not exceed team value
    model += (
        lpSum(
            player_vars[player_id]
            * predictions.loc[
                predictions["player_id"] == player_id, "current_cost"
            ].values[0]
            for player_id in predictions["player_id"]
        )
        <= team_budget,
        "Total_Cost_Constraint",
    )

    # Constraints: position limits
    for position, limit in position_constraints.items():
        model += (
            lpSum(
                player_vars[player_id]
                for player_id in predictions.loc[
                    predictions["position"] == position, "player_id"
                ]
            )
            <= limit,
            f"Position_{position}_Constraint",
        )

    # Constraints: total number of players must be 15
    model += (
        lpSum(player_vars[player_id] for player_id in predictions["player_id"]) == 15,
        "Total_Players_Constraint",
    )

    # Solve the problem
    model.solve()
    if LpStatus[model.status] != "Optimal":
        raise ValueError("No optimal solution found for the given constraints.")
    # Extract the selected players as a dict of player IDs and their predicted points
    selected_players = [
        player_id
        for player_id in predictions["player_id"]
        if player_vars[player_id].value() == 1
    ]
    selected_players = {
        player_id: predictions.loc[
            predictions["player_id"] == player_id, "predicted_points"
        ].values[0]
        for player_id in selected_players
    }
    return selected_players


def get_best_xi(best_15) -> dict:
    """
    Get the best XI from the best 15 players.
    Args:
        best_15 (dict): A dictionary with player IDs as keys and their predicted points as values.
    Returns:
        dict: A dictionary with player IDs as keys and their predicted points as values for the best XI.
    """
    # Sort players by predicted points
    min_positions = {
        1: 1,  # Goalkeepers
        2: 3,  # Defenders
        3: 3,  # Midfielders
        4: 1,  # Forwards
    }

    positions_df = pd.read_csv(get_data_path(SEASON, "players_season_data.csv"))
    positions_df = positions_df[["player_id", "element_type"]]
    positions_df.rename(columns={"element_type": "position"}, inplace=True)

    model = LpProblem(name="fpl_best_xi", sense=LpMaximize)
    player_vars = LpVariable.dict(
        "player",
        best_15.keys(),
        cat="Binary",
    )

    # Objective function: maximize the sum of predicted points
    model += (
        lpSum(
            player_vars[player_id] * best_15[player_id] for player_id in best_15.keys()
        ),
        "Total_Predicted_Points",
    )

    # Constraints: position limits
    for position, limit in min_positions.items():
        model += (
            lpSum(
                player_vars[player_id]
                for player_id in best_15.keys()
                if positions_df.loc[
                    positions_df["player_id"] == player_id, "position"
                ].values[0]
                == position
            )
            >= limit,
            f"Position_{position}_Constraint",
        )

    # Constraints: total number of players must be 11
    model += (
        lpSum(player_vars[player_id] for player_id in best_15.keys()) == 11,
        "Total_Players_Constraint",
    )

    # Solve the problem
    model.solve()

    if LpStatus[model.status] != "Optimal":
        raise ValueError("No optimal solution found for the given constraints.")

    # Extract the selected players as a dict of player IDs and their predicted points
    selected_players = [
        player_id for player_id in best_15.keys() if player_vars[player_id].value() == 1
    ]
    selected_players = {player_id: best_15[player_id] for player_id in selected_players}
    return selected_players


def main():
    gameweek = 38
    team_budget = 100.0  # Example team budget, adjust as needed
    best_team = get_best_team(gameweek, team_budget)
    print(f"Best 15 for gameweek {gameweek} with team budget {team_budget}:")

    for player_id, predicted_points in best_team.items():
        print(
            f"Player: {get_player_name(player_id)}, Predicted Points: {predicted_points}"
        )

    best_xi = get_best_xi(best_team)
    positions_df = pd.read_csv(get_data_path(SEASON, "players_season_data.csv"))
    positions_df = positions_df[["player_id", "element_type"]]
    positions_df.rename(columns={"element_type": "position"}, inplace=True)

    print("\nBest XI:")
    # Sort players into positions

    goalkeepers = [
        player_id
        for player_id in best_xi.keys()
        if positions_df.loc[positions_df["player_id"] == player_id, "position"].values[
            0
        ]
        == 1
    ]
    defenders = [
        player_id
        for player_id in best_xi.keys()
        if positions_df.loc[positions_df["player_id"] == player_id, "position"].values[
            0
        ]
        == 2
    ]
    midfielders = [
        player_id
        for player_id in best_xi.keys()
        if positions_df.loc[positions_df["player_id"] == player_id, "position"].values[
            0
        ]
        == 3
    ]
    forwards = [
        player_id
        for player_id in best_xi.keys()
        if positions_df.loc[positions_df["player_id"] == player_id, "position"].values[
            0
        ]
        == 4
    ]

    print("Goalkeepers:")
    for player_id in goalkeepers:
        print(
            f"Player: {get_player_name(player_id)}, Predicted Points: {best_xi[player_id]:.2f}"
        )
    print("\nDefenders:")
    for player_id in defenders:
        print(
            f"Player: {get_player_name(player_id)}, Predicted Points: {best_xi[player_id]:.2f}"
        )
    print("\nMidfielders:")
    for player_id in midfielders:
        print(
            f"Player: {get_player_name(player_id)}, Predicted Points: {best_xi[player_id]:.2f}"
        )
    print("\nForwards:")
    for player_id in forwards:
        print(
            f"Player: {get_player_name(player_id)}, Predicted Points: {best_xi[player_id]:.2f}"
        )


if __name__ == "__main__":
    main()
