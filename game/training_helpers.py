import numpy as np
from utils import distance
from simulation import simulate_game, calculate_fitness
from models import create_player_model, create_drone_model

# --- Parallel Evaluation Helper ---
def evaluate_candidate(args):
    index, player_weights, drone_weights, dt_sim, max_time, level = args
    p_model = create_player_model()
    p_model.set_weights(player_weights)
    d_model = create_drone_model()
    d_model.set_weights(drone_weights)
    return simulate_game(p_model, d_model, dt_sim, max_time)


def flatten_weights(weights_list):
    """Flatten a list of numpy arrays into a single 1D numpy array."""
    return np.concatenate([w.flatten() for w in weights_list])


def unflatten_weights(flat_vector, weights_template):
    """
    Convert a flat vector into a list of arrays with shapes matching weights_template.

    weights_template is a list of numpy arrays (typically from model.get_weights()).
    """
    new_weights = []
    index = 0
    for w in weights_template:
        size = w.size
        new_w = flat_vector[index:index + size].reshape(w.shape)
        new_weights.append(new_w)
        index += size
    return new_weights