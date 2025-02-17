import tensorflow as tf

def create_player_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(12,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(2, activation='tanh')
    ])
    return model

def create_drone_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(12,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(2, activation='tanh')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
