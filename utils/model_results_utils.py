import os
import re

import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_CONFIGS_FILE = os.path.join(_PROJECT_ROOT, "model_configs.csv")


def update_model_results_ranking():
    # Place a ranking for each metric in the results file
    try:
        with open("model_results.md", "r", encoding="utf-8") as f:
            content = f.read()
        # First, remove any existing rankings to avoid duplication
        content = re.sub(r"(### Test MAE: [\d.]+)( \(Rank: \d+\))+", r"\1", content)
        content = re.sub(
            r"(### Test Loss \(MSE\): [\d.]+)( \(Rank: \d+\))+", r"\1", content
        )

        # Extract all Test MAE and Test Loss values using regex
        mae_pattern = r"### Test MAE: ([\d.]+)"
        loss_pattern = r"### Test Loss \(MSE\): ([\d.]+)"

        mae_matches = re.findall(mae_pattern, content)
        loss_matches = re.findall(loss_pattern, content)

        # Convert to float and create lists with original values and indices
        mae_values = [(float(val), i) for i, val in enumerate(mae_matches)]
        loss_values = [(float(val), i) for i, val in enumerate(loss_matches)]

        # Sort by value (lower is better for both MAE and loss)
        mae_sorted = sorted(mae_values, key=lambda x: x[0])
        loss_sorted = sorted(loss_values, key=lambda x: x[0])

        # Create ranking dictionaries
        mae_rankings = {}
        loss_rankings = {}

        for rank, (value, original_index) in enumerate(mae_sorted, 1):
            mae_rankings[original_index] = rank

        for rank, (value, original_index) in enumerate(loss_sorted, 1):
            loss_rankings[original_index] = rank

        # Update content with rankings
        updated_content = content
        # Replace MAE entries with rankings
        mae_matches_with_positions = list(re.finditer(mae_pattern, content))
        for i, match in enumerate(reversed(mae_matches_with_positions)):
            original_index = len(mae_matches_with_positions) - 1 - i
            ranking = mae_rankings[original_index]
            new_text = f"### Test MAE: {mae_matches[original_index]} (Rank: {ranking})"
            updated_content = (
                updated_content[: match.start()]
                + new_text
                + updated_content[match.end() :]
            )

        # Replace Loss entries with rankings
        loss_matches_with_positions = list(re.finditer(loss_pattern, updated_content))
        for i, match in enumerate(reversed(loss_matches_with_positions)):
            original_index = len(loss_matches_with_positions) - 1 - i
            ranking = loss_rankings[original_index]
            new_text = (
                f"### Test Loss (MSE): {loss_matches[original_index]} (Rank: {ranking})"
            )
            updated_content = (
                updated_content[: match.start()]
                + new_text
                + updated_content[match.end() :]
            )
        # Write updated content back to file
        with open("model_results.md", "w", encoding="utf-8") as f:
            f.write(updated_content)

        print("\n---------Update model rankings in model_results.md----------\n")
        print("Rankings updated successfully!")
        print(f"Total models: {len(mae_matches)}")
        if mae_matches:
            best_mae = min(mae_values)
            best_loss = min(loss_values)
            print(f"Best MAE: {best_mae[0]:.4f} (Model {mae_rankings[best_mae[1]]})")
            print(
                f"Best Loss: {best_loss[0]:.4f} (Model {loss_rankings[best_loss[1]]})"
            )

    except Exception as e:
        print(f"Error updating rankings: {e}")


def get_model_layers_info(model):
    layers_info = []
    for i, layer in enumerate(model.layers, start=1):
        layer_dict = {
            "name": layer.name,
            "type": type(layer).__name__,
            "input_shape": str(getattr(layer, "input_shape", None)),
            "output_shape": str(getattr(layer, "output_shape", None)),
            "activation": getattr(layer, "activation", None),
        }
        layer_dict["activation"] = (
            layer_dict["activation"].__name__ if layer_dict["activation"] else None
        )
        layers_info.append(layer_dict)

    layers_info_str = "\n".join(
        f"Layer {i}: {layer['name']} ({layer['type']}) - "
        f"Input shape: {layer['input_shape']}, "
        f"Output shape: {layer['output_shape']}, "
        f"Activation: {layer['activation']}"
        for i, layer in enumerate(layers_info, start=1)
    )
    return layers_info, layers_info_str


def write_model_config(config: dict, model):
    current_config = pd.DataFrame([config])
    try:
        model_configs = pd.read_csv(MODEL_CONFIGS_FILE, dtype=str)
    except FileNotFoundError:
        model_configs = pd.DataFrame(columns=current_config.columns)

    layers_info, layers_info_str = get_model_layers_info(model)
    current_config["layers_info"] = layers_info_str
    model_configs = pd.concat([model_configs, current_config], ignore_index=True)
    model_configs.to_csv(MODEL_CONFIGS_FILE, index=False, encoding="utf-8")


def check_duplicate_config(config: dict, model):
    """
    Check if the current model configuration already exists in the model_configs.csv file.
    Returns True if a duplicate is found, False otherwise.
    """
    try:
        model_configs = pd.read_csv(MODEL_CONFIGS_FILE, dtype=str)
    except FileNotFoundError:
        return False  # No previous configs, so no duplicates

    # Check if the current config already exists in the model_configs
    current_config = pd.DataFrame([config])
    layers_info, layers_info_str = get_model_layers_info(model)
    current_config["layers_info"] = layers_info_str

    if set(model_configs.columns) != set(current_config.columns):
        raise ValueError(
            "Model configs file has different columns than the current config."
        )

    cols = list(current_config.columns)

    for _, row in model_configs.iterrows():
        for col in cols:
            if str(row[col]) != str(current_config[col].values[0]):
                return False  # Found a mismatch in any column

    return True
