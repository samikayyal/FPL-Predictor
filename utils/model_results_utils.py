import os
import re

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_CONFIGS_FILE = os.path.join(_PROJECT_ROOT, "model_configs.csv")


def update_model_results_ranking():
    # Place a ranking for each metric in the results file (excluding MSE from average)
    try:
        with open("model_results.md", "r", encoding="utf-8") as f:
            content = f.read()

        # Remove any existing rankings to avoid duplication
        content = re.sub(
            r"(### Test Loss \(Huber\): [\d.]+)( \(Rank: \d+\))+", r"\1", content
        )
        content = re.sub(
            r"(### Test Log-Cosh: [\d.]+)( \(Rank: \d+\))+", r"\1", content
        )
        content = re.sub(r"(### Test MSE: [\d.]+)( \(Rank: \d+\))+", r"\1", content)
        content = re.sub(r"(### Test MAE: [\d.]+)( \(Rank: \d+\))+", r"\1", content)

        # Extract all values using regex
        huber_pattern = r"### Test Loss \(Huber\): ([\d.]+)"
        mse_pattern = r"### Test MSE: ([\d.]+)"
        logcosh_pattern = r"### Test Log-Cosh: ([\d.]+)"
        mae_pattern = r"### Test MAE: ([\d.]+)"

        huber_matches = re.findall(huber_pattern, content)
        mse_matches = re.findall(mse_pattern, content)
        logcosh_matches = re.findall(logcosh_pattern, content)
        mae_matches = re.findall(mae_pattern, content)

        # lists with original values and indices
        huber_values = [(float(val), i) for i, val in enumerate(huber_matches)]
        mse_values = [(float(val), i) for i, val in enumerate(mse_matches)]
        logcosh_values = [(float(val), i) for i, val in enumerate(logcosh_matches)]
        mae_values = [(float(val), i) for i, val in enumerate(mae_matches)]

        # Sort by value (lower is better for all)
        huber_sorted = sorted(huber_values, key=lambda x: x[0])
        mse_sorted = sorted(mse_values, key=lambda x: x[0])
        logcosh_sorted = sorted(logcosh_values, key=lambda x: x[0])
        mae_sorted = sorted(mae_values, key=lambda x: x[0])

        mae_rankings = {}
        mse_rankings = {}
        logcosh_rankings = {}
        huber_rankings = {}

        for rank, (value, original_index) in enumerate(mae_sorted, 1):
            mae_rankings[original_index] = rank
        for rank, (value, original_index) in enumerate(mse_sorted, 1):
            mse_rankings[original_index] = rank
        for rank, (value, original_index) in enumerate(logcosh_sorted, 1):
            logcosh_rankings[original_index] = rank
        for rank, (value, original_index) in enumerate(huber_sorted, 1):
            huber_rankings[original_index] = rank

        # Compute average rank (excluding MSE)
        avg_ranks = {}
        n_models = len(mae_matches)

        for i in range(n_models):
            avg_ranks[i] = np.mean(
                [
                    mae_rankings.get(i, n_models + 1),
                    logcosh_rankings.get(i, n_models + 1),
                    huber_rankings.get(i, n_models + 1),
                ]
            )

        # Create a fresh copy of content for updating with rankings
        updated_content = content

        # Process each model section to add rankings
        # We'll split by model sections to avoid issues with overlapping regex replacements
        model_sections = re.split(r"(# Model Training Results)", updated_content)
        result_content = ""

        # Process the sections one by one
        for i in range(len(model_sections)):
            section = model_sections[i]

            # For actual model section content (not the headers)
            if i > 0 and i % 2 == 0:
                section_index = (i // 2) - 1

                if section_index < n_models:
                    # Replace metrics with rankings in this section

                    # Huber
                    huber_match = re.search(huber_pattern, section)
                    if huber_match:
                        old_huber = huber_match.group(0)
                        new_huber = f"### Test Loss (Huber): {huber_matches[section_index]} (Rank: {huber_rankings[section_index]})"
                        section = section.replace(old_huber, new_huber)

                    # MSE
                    mse_match = re.search(mse_pattern, section)
                    if mse_match:
                        old_mse = mse_match.group(0)
                        new_mse = f"### Test MSE: {mse_matches[section_index]} (Rank: {mse_rankings[section_index]})"
                        section = section.replace(old_mse, new_mse)

                    # LogCosh
                    logcosh_match = re.search(logcosh_pattern, section)
                    if logcosh_match:
                        old_logcosh = logcosh_match.group(0)
                        new_logcosh = f"### Test Log-Cosh: {logcosh_matches[section_index]} (Rank: {logcosh_rankings[section_index]})"
                        section = section.replace(old_logcosh, new_logcosh)

                    # MAE
                    mae_match = re.search(mae_pattern, section)
                    if mae_match:
                        old_mae = mae_match.group(0)
                        new_mae = f"### Test MAE: {mae_matches[section_index]} (Rank: {mae_rankings[section_index]})"
                        section = section.replace(old_mae, new_mae)

            result_content += section

        # Write updated content back to file
        with open("model_results.md", "w", encoding="utf-8") as f:
            f.write(result_content)

        print("\n---------Update model rankings in model_results.md----------\n")
        print("Rankings updated successfully!")
        print(f"Total models: {n_models}")
        if mae_matches:
            best_avg = min(avg_ranks.items(), key=lambda x: x[1])
            print(f"Best average rank: {best_avg[1]:.2f} (Model {best_avg[0]+1})")
    except Exception as e:
        print(f"Error updating rankings: {e}")


def sort_models_by_average_rank():
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

        # attatch the header back to each section
        model_sections = [
            "# Model Training Results" + section for section in model_sections
        ]

        # Dictionary to store model sections with their average ranks
        model_ranks = {}

        # Extract ranks for each model
        huber_rank_pattern = r"### Test Loss \(Huber\): [\d.]+ \(Rank: (\d+)\)"
        mae_rank_pattern = r"### Test MAE: [\d.]+ \(Rank: (\d+)\)"
        logcosh_rank_pattern = r"### Test Log-Cosh: [\d.]+ \(Rank: (\d+)\)"

        # Process each model section
        for section in model_sections:
            try:
                mae_rank_match = re.search(mae_rank_pattern, section)
                logcosh_rank_match = re.search(logcosh_rank_pattern, section)
                huber_rank_match = re.search(huber_rank_pattern, section)

                if mae_rank_match and logcosh_rank_match and huber_rank_match:
                    mae_rank = int(mae_rank_match.group(1))
                    logcosh_rank = int(logcosh_rank_match.group(1))
                    huber_rank = int(huber_rank_match.group(1))
                    avg_rank = (mae_rank + logcosh_rank + huber_rank) / 3
                    model_ranks[section] = avg_rank
                else:
                    print("Warning: Could not find all ranks in a model section")
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

        print("\nModels in model_results.md sorted by average rank (excluding MSE).")
        print(f"Total models: {len(sorted_models)}")
        if sorted_models:
            best_model = sorted_models[0]
            mae_rank_match = re.search(mae_rank_pattern, best_model[0])
            logcosh_rank_match = re.search(logcosh_rank_pattern, best_model[0])
            huber_rank_match = re.search(huber_rank_pattern, best_model[0])
            if mae_rank_match and logcosh_rank_match and huber_rank_match:
                mae_rank = int(mae_rank_match.group(1))
                logcosh_rank = int(logcosh_rank_match.group(1))
                huber_rank = int(huber_rank_match.group(1))
                print(
                    f"Best model has average rank: {best_model[1]:.2f} (MAE rank: {mae_rank}, Log-Cosh rank: {logcosh_rank}, Huber rank: {huber_rank})"
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
    sort_models_by_average_rank()
