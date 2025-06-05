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


def sort_models_by_average_rank():
    """
    Sort models in model_results.md based on average of MAE and MSE ranks and rewrite the file.
    Lower average rank is better.
    """
    try:
        with open("model_results.md", "r", encoding="utf-8") as f:
            content = f.read()

        # Extract model sections using regex
        model_sections = re.split(r"# Model Training Results", content)
        if len(model_sections) <= 1:
            print("No model sections found in model_results.md")
            return

        # Save the header (content before the first model)
        header = ""
        if model_sections[0].strip():
            header = model_sections[0]
        model_sections.pop(0)

        # Prepend the header back to each section
        model_sections = [
            "# Model Training Results" + section for section in model_sections
        ]

        # Dictionary to store model sections with their average ranks
        model_ranks = {}

        # Extract ranks for each model
        mae_rank_pattern = r"### Test MAE: [\d.]+ \(Rank: (\d+)\)"
        mse_rank_pattern = r"### Test Loss \(MSE\): [\d.]+ \(Rank: (\d+)\)"

        # Process each model section
        for section in model_sections:
            try:
                mae_rank_match = re.search(mae_rank_pattern, section)
                mse_rank_match = re.search(mse_rank_pattern, section)

                if mae_rank_match and mse_rank_match:
                    mae_rank = int(mae_rank_match.group(1))
                    mse_rank = int(mse_rank_match.group(1))
                    avg_rank = (mae_rank + mse_rank) / 2

                    # Store the section with its average rank
                    model_ranks[section] = avg_rank
                else:
                    print("Warning: Could not find ranks in a model section")
            except Exception as e:
                print(f"Error processing model section: {e}")

        # Sort models by average rank (lower is better)
        sorted_models = sorted(model_ranks.items(), key=lambda x: x[1])

        # Combine the sorted sections
        sorted_content = header
        for model_section, _ in sorted_models:
            sorted_content += model_section

        # Write the sorted content back to file
        with open("model_results.md", "w", encoding="utf-8") as f:
            f.write(sorted_content)

        print("\nModels in model_results.md sorted by average rank.")
        print(f"Total models: {len(sorted_models)}")
        if sorted_models:
            best_model = sorted_models[0]
            mae_rank_match = re.search(mae_rank_pattern, best_model[0])
            mse_rank_match = re.search(mse_rank_pattern, best_model[0])
            if mae_rank_match and mse_rank_match:
                mae_rank = int(mae_rank_match.group(1))
                mse_rank = int(mse_rank_match.group(1))
                print(
                    f"Best model has average rank: {best_model[1]:.2f} (MAE rank: {mae_rank}, MSE rank: {mse_rank})"
                )

    except Exception as e:
        print(f"Error sorting models by average rank: {e}")


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


if __name__ == "__main__":
    update_model_results_ranking()
