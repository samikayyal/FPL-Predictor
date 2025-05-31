import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout

# ====================
# Some preperation
# ====================

print("Preparing the data...")
df = pd.read_csv("final_df.csv")
df = df.sort_values(by=["gw", "player_id"], ascending=True)

# Drop the id columns until i find a better way to handle them
df = df.drop(columns=["player_id", "team_id", "opponent_team_id"])

# convert bool columns to int
df = df.apply(lambda x: x.astype(int) if x.dtype == "bool" else x)

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
print("Standardizing the data...")
scaler = StandardScaler()

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

# Define the model
model = keras.models.Sequential(
    [
        Dense(256, activation="relu", input_shape=(X_train_processed.shape[1],)),
        Dropout(0.2),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="linear"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss="mean_squared_error",  # Common for regression
    metrics=["mean_absolute_error"],  # Optional: another metric to track
)

# Print model summary
model.summary()

# Train the model
# Optional: Early stopping to prevent overfitting
callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", monitor="val_loss", save_best_only=True
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
    epochs=100,
    validation_data=(X_val_processed, y_val),
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test_processed, y_test, verbose=0)
print(f"\nTest MAE: {test_mae:.4f}")
print(f"Test Loss (MSE): {test_loss:.4f}")

# To make predictions:
# predictions = model.predict(X_test_processed)
