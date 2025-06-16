import io
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras
from tensorflow.keras.layers import (  # type: ignore
    BatchNormalization,
    Dense,
    Dropout,
    Input,
)

from utils.model_results_utils import (
    check_duplicate_config,
    sort_models_by_average_rank,
    update_model_results_ranking,
    write_model_config,
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ====================
# Some preperation
# ====================
__start_time = time.time()
print("------- Preparing the data...")
df = pd.read_csv("final_df.csv")
# Store ids for later use
player_ids = df["player_id"].copy()

df = df.sort_values(by=["gw", "player_id"], ascending=True)

# Drop the id columns until i find a better way to handle them
df = df.drop(columns=["player_id", "opponent_team_id"])

# convert bool columns to int
df = df.apply(lambda x: x.astype(int) if x.dtype == "bool" else x)

# =========================
# Trying to manipulate the data to improve the model
# =========================

# df = df[df.minutes_season > 100]  # Filter out players with less than 100 minutes played

# One hot encode team column (if i remove one hot then this drop it from the data)
team_dummies = pd.get_dummies(df["team_id"], prefix="team")
df = pd.concat([df, team_dummies], axis=1)

# Drop the original team_id column
df = df.drop(columns=["team_id"])

# =========================
# Scaling and Splitting the Data
# =========================

# Split the data into 70% train, 15% validation, and 15% test sets
n_gws = int(df.gw.nunique())
train_end = int(n_gws * 0.7)
val_end = int(n_gws * 0.85)

train_df = df[df.gw <= train_end]
val_df = df[(df.gw > train_end) & (df.gw <= val_end)]
test_df = df[df.gw > val_end]

print(
    f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}"
)

# Separate features and target
X_train = train_df.drop(columns=["total_points"])
y_train = train_df["total_points"]
X_val = val_df.drop(columns=["total_points"])
y_val = val_df["total_points"]
X_test = test_df.drop(columns=["total_points"])
y_test = test_df["total_points"]

# Standardize the data
print("\n------- Standardizing the data...")
# try standard scaling and min-max scaling
scaler = MinMaxScaler()  # You can switch to StandardScaler() if needed
# scaler = StandardScaler()


cols_to_scale = [col for col in X_train.columns if col != "gw"]

# Fit the scaler on the training data features (excluding 'gw')
scaler.fit(X_train[cols_to_scale])

# Transform the features for train, validation, and test sets
X_train_scaled = scaler.transform(X_train[cols_to_scale])
X_val_scaled = scaler.transform(X_val[cols_to_scale])
X_test_scaled = scaler.transform(X_test[cols_to_scale])

# Convert scaled arrays back to DataFrames
X_train_scaled_df = pd.DataFrame(
    X_train_scaled, columns=cols_to_scale, index=X_train.index
)
X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=cols_to_scale, index=X_val.index)
X_test_scaled_df = pd.DataFrame(
    X_test_scaled, columns=cols_to_scale, index=X_test.index
)

# Combine scaled features with the unscaled 'gw' column
X_train_processed = pd.concat([X_train[["gw"]], X_train_scaled_df], axis=1)
X_val_processed = pd.concat([X_val[["gw"]], X_val_scaled_df], axis=1)
X_test_processed = pd.concat([X_test[["gw"]], X_test_scaled_df], axis=1)

print(f"X_train_processed shape: {X_train_processed.shape}")
print(f"X_val_processed shape: {X_val_processed.shape}")
print(f"X_test_processed shape: {X_test_processed.shape}")

# ================================
# Neural Network Model Definition
# ================================

# ==== Hyperparameters ====
LEARNING_RATE: float = 0.0003
EPOCHS: int = 100
BATCH_SIZE: int = 64
DROPOUT_RATE: float = 0.4
IS_L2_REGULARIZATION: bool = True
L2_REGULARIZATION_RATE: float = 1e-04 if IS_L2_REGULARIZATION else 0.0
HUBER_DELTA: float = 1.5

# Define the model
model = keras.models.Sequential(
    [
        Input(shape=(X_train_processed.shape[1],)),
        Dense(
            128,
            activation="relu",
            kernel_regularizer=(
                keras.regularizers.l2(L2_REGULARIZATION_RATE)
                if IS_L2_REGULARIZATION
                else None
            ),
        ),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        Dense(
            64,
            activation="relu",
            kernel_regularizer=(
                keras.regularizers.l2(L2_REGULARIZATION_RATE)
                if IS_L2_REGULARIZATION
                else None
            ),
        ),
        # BatchNormalization(),
        # Dropout(DROPOUT_RATE),
        # Dense(
        #     64,
        #     activation="relu",
        #     kernel_regularizer=(
        #         keras.regularizers.l2(L2_REGULARIZATION_RATE)
        #         if IS_L2_REGULARIZATION
        #         else None
        #     ),
        # ),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        Dense(
            32,
            activation="relu",
            kernel_regularizer=(
                keras.regularizers.l2(L2_REGULARIZATION_RATE)
                if IS_L2_REGULARIZATION
                else None
            ),
        ),
        Dense(1, activation="linear"),
    ]
)


# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=keras.losses.Huber(delta=HUBER_DELTA),
    metrics=[
        "mean_squared_error",
        keras.losses.LogCosh(),
        "mean_absolute_error",
    ],
)


# Print model summary
model_summary_io = io.StringIO()
model.summary(print_fn=lambda x: model_summary_io.write(x + "\n"))
model_summary_string = model_summary_io.getvalue()
model_summary_io.close()


print("------- Model Summary:")
print(model_summary_string)

# Write config to a json file to keep track of the hyperparameters and not include duplicates
config = {
    "LEARNING_RATE": LEARNING_RATE,
    "EPOCHS": EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "DROPOUT_RATE": DROPOUT_RATE,
    "IS_L2_REGULARIZATION": IS_L2_REGULARIZATION,
    "L2_REGULARIZATION_RATE": L2_REGULARIZATION_RATE,
    "HUBER_DELTA": HUBER_DELTA,
}

if check_duplicate_config(config, model):
    print(
        "------- Warning: Duplicate configuration detected! Not training the model again."
    )
    print("This configuration already exists in the model_configs.csv file.")
    exit(0)  # Exit if the configuration already exists

write_model_config(config, model)


# Train the model
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.keras", monitor="val_loss", save_best_only=True
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=6, min_lr=1e-6
    ),
]

history = model.fit(
    X_train_processed,
    y_train,
    epochs=EPOCHS,
    validation_data=(X_val_processed, y_val),
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

# Evaluate the model on the test set
test_loss, test_mse, test_log_cosh, test_mae = model.evaluate(
    X_test_processed, y_test, verbose=1
)
print(f"\n\nTest Loss (Huber): {test_loss:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test Log-Cosh: {test_log_cosh:.4f}")
print(f"Test MAE: {test_mae:.4f}")


# Save state and results
__end_time = time.time()
with open("model_results.md", "a", encoding="utf-8") as f:  # Changed to .md
    f.write("# Model Training Results\n\n")
    f.write(f"Time to train: {__end_time - __start_time:.2f} seconds\n\n")

    f.write("## Hyperparameters\n")
    f.write(f"- LEARNING_RATE: {LEARNING_RATE}\n")
    f.write(f"- EPOCHS: {EPOCHS}\n")
    f.write(f"- BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"- DROPOUT_RATE: {DROPOUT_RATE}\n\n")
    f.write(f"- IS_L2_REGULARIZATION: {IS_L2_REGULARIZATION}\n")
    f.write(f"- L2_REGULARIZATION_RATE: {L2_REGULARIZATION_RATE}\n\n")
    f.write(f"- HUBER_DELTA: {HUBER_DELTA}\n\n")

    f.write("## Evaluation Metrics\n")
    f.write(f"### Test Loss (Huber): {test_loss:.4f}\n")
    f.write(f"### Test MSE: {test_mse:.4f}\n")
    f.write(f"### Test Log-Cosh: {test_log_cosh:.4f}\n")
    f.write(f"### Test MAE: {test_mae:.4f}\n\n")

    f.write("## Model Architecture\n")
    f.write(f"```\n{model_summary_string}```\n")  # Use Markdown code block for summary

# Plots
plt.figure(figsize=(15, 10))  # Adjusted figure size for 2x2 subplots

# First subplot: Loss (Mean Absolute Error) Over Epochs
plt.subplot(2, 2, 1)
plt.plot(history.history["loss"], label="Train Huber Loss")
plt.plot(history.history["val_loss"], label="Validation Huber Loss")
plt.title("Huber Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Huber Loss")
plt.legend()

# Second subplot: Mean Squared Error Over Epochs
plt.subplot(2, 2, 2)
plt.plot(history.history["mean_squared_error"], label="Train MSE")
plt.plot(history.history["val_mean_squared_error"], label="Validation MSE")
plt.title("Mean Squared Error Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.legend()

# Third subplot: Log-Cosh Over Epochs
plt.subplot(2, 2, 3)
# Corrected keys:
plt.plot(history.history["log_cosh"], label="Train Log-Cosh")
plt.plot(history.history["val_log_cosh"], label="Validation Log-Cosh")
plt.title("Log-Cosh Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Log-Cosh")
plt.legend()

# Fourth subplot: Huber Loss Over Epochs
plt.subplot(2, 2, 4)
# Corrected keys:
plt.plot(history.history["mean_absolute_error"], label="Train MAE")
plt.plot(history.history["val_mean_absolute_error"], label="Validation MAE")
plt.title("MAE Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()


plt.tight_layout()  # Adjusts subplot params for a tight layout
plot_filename = os.path.join(
    "model_plots",
    f'model_plots_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png',
)
if not os.path.exists("model_plots"):
    os.makedirs("model_plots")
plt.savefig(plot_filename)
plt.show()

with open("model_results.md", "a", encoding="utf-8") as f:
    f.write("## Training Plots\n")
    f.write(f"![Training Plots]({plot_filename})\n\n\n")

# Update rankings in the model results markdown file
update_model_results_ranking()
sort_models_by_average_rank()
