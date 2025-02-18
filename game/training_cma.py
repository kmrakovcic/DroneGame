import time
import concurrent.futures
import numpy as np
import cma
import os
from models import create_player_model, create_drone_model
from training_helpers import evaluate_candidate
from training_helpers import unflatten_weights, flatten_weights
from simulation import simulate_game

# --- Fitness Functions for Coevolution ---
def fitness_player(flat_player_weights, flat_drone_weights, dt, max_time):
    """Evaluate the player fitness when using these candidate weights.
       (Assume higher is better; we will return negative value for minimization.)"""
    # Create player and drone models.
    p_model = create_player_model()
    d_model = create_drone_model()
    p_template = p_model.get_weights()
    d_template = d_model.get_weights()

    p_model.set_weights(unflatten_weights(flat_player_weights, p_template))
    d_model.set_weights(unflatten_weights(flat_drone_weights, d_template))

    player_fitness, drone_fitness, sim_time = simulate_game(p_model, d_model, dt_sim=dt, max_time=max_time)
    return -player_fitness  # CMA-ES minimizes


def fitness_drone(flat_player_weights, flat_drone_weights, dt, max_time):
    """Evaluate the drone fitness using the given candidate weights.
       (Return negative fitness so that CMA-ES minimizes.)"""
    p_model = create_player_model()
    d_model = create_drone_model()
    p_template = p_model.get_weights()
    d_template = d_model.get_weights()

    p_model.set_weights(unflatten_weights(flat_player_weights, p_template))
    d_model.set_weights(unflatten_weights(flat_drone_weights, d_template))

    player_fitness, drone_fitness, sim_time = simulate_game(p_model, d_model, dt_sim=dt, max_time=max_time)
    return -drone_fitness  # CMA-ES minimizes


def evaluate_pairing(args):
    """
    Given a tuple (p_candidate, d_candidate, num_evals),
    run simulate_game num_evals times and return the average (player_fitness, drone_fitness).
    (Note: fitness functions return negative values so that CMA-ES minimizes.)
    """
    p_candidate, d_candidate, num_evals, dt, max_time= args
    player_fits = []
    drone_fits = []
    for _ in range(num_evals):
        pf = fitness_player(p_candidate, d_candidate, dt, max_time)
        df = fitness_drone(p_candidate, d_candidate, dt, max_time)
        player_fits.append(pf)
        drone_fits.append(df)
    return np.mean(player_fits), np.mean(drone_fits)


# === Genetic Algorithm Training Mode ===
def run_training(save_path, use_parallel_evaluation=True):
    n = 300  # population size
    dt_sim = 0.033  # 30 FPS
    max_time = 60.0
    max_generations = 1000
    if save_path[-1] != '/':
        save_path += '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        initial_player = flatten_weights(create_player_model().get_weights())
        initial_drone = flatten_weights(create_drone_model().get_weights())

    # Initialize two CMA-ES instances.
    else:
        if "drone.keras" in os.listdir(save_path):
            initial_drone = flatten_weights(create_drone_model().load(save_path+"drone.keras").get_weights())
        else:
            initial_drone = flatten_weights(create_drone_model().get_weights())
        if "player.keras" in os.listdir(save_path):
            initial_player = flatten_weights(create_player_model().load(save_path+"player.keras").get_weights())
        else:
            initial_player = flatten_weights(create_player_model().get_weights())
    player_sigma = 0.5
    drone_sigma = 0.5
    player_es = cma.CMAEvolutionStrategy(initial_player, player_sigma)
    drone_es  = cma.CMAEvolutionStrategy(initial_drone, drone_sigma)

    # Number of independent evaluations per candidate pair.
    num_evals_per_pair = 1

    for generation in range(max_generations):
        start_gen = time.time()
        # Ask for a batch of candidate solutions from each population.
        player_candidates = player_es.ask()
        drone_candidates  = drone_es.ask()

        # Create a list of candidate tuples.
        # Here, we pair candidate i from the player with candidate i from the drone.
        candidate_tuples = [
            (p_candidate, d_candidate, num_evals_per_pair, dt_sim, max_time)
            for p_candidate, d_candidate in zip(player_candidates, drone_candidates)
        ]

        # Evaluate all candidate pairs in parallel.
        if use_parallel_evaluation:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(evaluate_pairing, candidate_tuples))
        else:
            results = []
            for c, ct in enumerate(candidate_tuples):
                results.append(evaluate_pairing(ct))
                print ("\rCompleted candidate: "+str(c)+"/"+str(n), end="")

        # Unpack the results.
        player_fit_values = [res[0] for res in results]
        drone_fit_values  = [res[1] for res in results]

        # Update each CMA-ES instance with the averaged fitness values.
        player_es.tell(player_candidates, player_fit_values)
        drone_es.tell(drone_candidates, drone_fit_values)

        gen_time = time.time() - start_gen
        pop_size = len(player_candidates)
        print(f"Generation {generation + 1}: Time taken = {gen_time:.2f} sec, Population size = {pop_size}")
        print("  Best player fitness:", -1*min(player_fit_values))
        print("  Best drone  fitness:", -1*min(drone_fit_values))

        # Save the best models from this generation.
        best_player_weights = player_es.result.xbest
        best_drone_weights = drone_es.result.xbest
        best_player_model = create_player_model()
        best_drone_model = create_drone_model()
        best_player_model.set_weights(unflatten_weights(best_player_weights, best_player_model.get_weights()))
        best_drone_model.set_weights(unflatten_weights(best_drone_weights, best_drone_model.get_weights()))
        best_player_model.save(save_path+"player.keras")
        best_drone_model.save(save_path+f"drone.keras")

        if player_es.stop() or drone_es.stop():
            break

    print("Co-evolution complete!")