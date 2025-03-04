import tensorflow as tf

def create_player_hunter_model(input_shape=22):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='leaky_relu'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(32, activation='leaky_relu'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(16, activation='sigmoid'),
        tf.keras.layers.Dense(8, activation='leaky_relu'),
        tf.keras.layers.Dense(4, activation='leaky_relu'),
        tf.keras.layers.Dense(2, activation='tanh')
    ])
    return model

def create_drone_hunter_model(input_shape=22):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='leaky_relu'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(32, activation='leaky_relu'),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Dense(16, activation='sigmoid'),
        tf.keras.layers.Dense(8, activation='leaky_relu'),
        tf.keras.layers.Dense(4, activation='leaky_relu'),
        tf.keras.layers.Dense(2, activation='tanh')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
