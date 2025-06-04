import pandas as pd
import tensorflow as tf

model = tf.keras.models.load_model("best_model.keras")

def predict_points(player_id:int, gameweek:int):
    