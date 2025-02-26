import numpy as np
import time
import concurrent.futures
import os
from models import create_player_hunter_model, create_drone_hunter_model
from training_helpers import evaluate_candidate
from training_helpers import unflatten_weights, flatten_weights
from simulation import simulate_game


def fitness_player_drone(flat_player_weights, flat_drone_weights, dt, max_time):
    """Evaluate the player fitness when using these candidate weights.
       (Assume higher is better; we will return negative value for minimization.)"""
    # Create player and drone models.
    p_model = create_player_hunter_model()
    d_model = create_drone_hunter_model()
    p_template = p_model.get_weights()
    d_template = d_model.get_weights()

    p_model.set_weights(unflatten_weights(flat_player_weights, p_template))
    d_model.set_weights(unflatten_weights(flat_drone_weights, d_template))

    player_fitness, drone_fitness, sim_time = simulate_game(p_model, d_model, dt_sim=dt, max_time=max_time)
    return -player_fitness, -drone_fitness


# ================================================================
# NES Optimizer (Simple Isotropic Version)
# ================================================================
class NES:
    def __init__(self, initial_mean, sigma, popsize, lr_mu=0.1, lr_sigma=0.1):
        """
        initial_mean: 1D numpy array representing the initial parameters.
        sigma: initial standard deviation (scalar).
        popsize: number of candidate solutions per generation.
        lr_mu: learning rate for updating the mean.
        lr_sigma: learning rate for updating the standard deviation.
        """
        self.mu = np.array(initial_mean, copy=True)
        self.sigma = sigma
        self.popsize = popsize
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.best_candidate = None
        self.best_fitness = np.inf  # since we are minimizing (fitness values are negated)

    def ask(self):
        # Sample noise for each candidate
        self.noise = np.random.randn(self.popsize, self.mu.size)
        # Candidates are given by: x = mu + sigma * noise
        samples = self.mu + self.sigma * self.noise
        # Return as list of vectors
        return np.array(samples.tolist())

    def tell(self, candidates, fitnesses):
        """
        candidates: list of candidate vectors (each a 1D array)
        fitnesses: list (or array) of scalar fitness values (remember, lower is better here)
        """
        fitnesses = np.array(fitnesses)
        # Natural gradient estimate for mu: average over (fitness * noise)
        grad_mu = np.dot(fitnesses, self.noise) / self.popsize
        self.mu += self.lr_mu * self.sigma * grad_mu

        # For sigma update, we use a simple rule:
        # Compute gradient for sigma: average over fitness * (noise^2 - 1)
        grad_sigma = np.mean(fitnesses[:, None] * (self.noise ** 2 - 1), axis=0)
        # Here we update sigma multiplicatively (averaging the gradient components)
        self.sigma *= np.exp((self.lr_sigma / 2.0) * grad_sigma.mean())

        # Keep track of the best candidate seen in this generation.
        best_index = np.argmin(fitnesses)
        self.best_candidate = candidates[best_index]
        self.best_fitness = fitnesses[best_index]

    def stop(self):
        # For simplicity, we never auto-stop in this example.
        return False

    def result(self):
        return self.best_candidate, self.best_fitness


def evaluate_pairing(args):
    """
            Given a tuple (p_candidate, d_candidate, num_evals, dt, max_time),
            run simulate_game num_evals times and return the average (player_fitness, drone_fitness).
            (Fitness values are negated so that lower is better.)
            """
    p_candidate, d_candidate, num_evals, dt, max_time = args
    player_fits = []
    drone_fits = []
    for _ in range(num_evals):
        pf, df = fitness_player_drone(p_candidate, d_candidate, dt, max_time)
        player_fits.append(pf)
        drone_fits.append(df)
    return np.mean(player_fits), np.mean(drone_fits)


# ================================================================
# Replace CMA-ES with NES in your co-evolution loop
# ================================================================
def run_training(save_path, epochs, use_parallel_evaluation=True):
    n = 300  # population size
    dt_sim = 0.033  # simulation time step (30 FPS)
    max_time = 60.0
    max_generations = epochs

    # Ensure save_path ends with '/'
    if save_path[-1] != '/':
        save_path += '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        initial_player = flatten_weights(create_player_hunter_model().get_weights())
        initial_drone = flatten_weights(create_drone_hunter_model().get_weights())
    else:
        if "drone.keras" in os.listdir(save_path):
            import tensorflow as tf
            initial_drone = flatten_weights(tf.keras.models.load_model(save_path + "drone.keras").get_weights())
        else:
            initial_drone = flatten_weights(create_drone_hunter_model().get_weights())
        if "player.keras" in os.listdir(save_path):
            import tensorflow as tf
            initial_player = flatten_weights(tf.keras.models.load_model(save_path + "player.keras").get_weights())
        else:
            initial_player = flatten_weights(create_player_hunter_model().get_weights())

    # Set initial sigma values for NES
    player_sigma = 0.5
    drone_sigma = 0.5

    # Initialize NES optimizers for player and drone.
    player_nes = NES(initial_player, player_sigma, popsize=n, lr_mu=0.1, lr_sigma=0.1)
    drone_nes = NES(initial_drone, drone_sigma, popsize=n, lr_mu=0.1, lr_sigma=0.1)

    # Number of independent simulation evaluations per candidate pair.
    num_evals_per_pair = 1

    for generation in range(max_generations):
        start_gen = time.time()
        # Get candidate solutions from NES optimizers.
        player_candidates = player_nes.ask()
        drone_candidates = drone_nes.ask()

        # Pair candidate i from player with candidate i from drone.
        candidate_tuples = [
            (p_candidate, d_candidate, num_evals_per_pair, dt_sim, max_time)
            for p_candidate, d_candidate in zip(player_candidates, drone_candidates)
        ]

        # Evaluate candidate pairs in parallel if desired.
        if use_parallel_evaluation:
            restarts = 0
            while True:
                try:
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        results = list(executor.map(evaluate_pairing, candidate_tuples, timeout=600))
                        break
                except concurrent.futures.TimeoutError:
                    if restarts <= 5:
                        restarts += 1
                        print(f"Timeout occurred in epoch {epoch + 1}! Restarting the epoch...")
                    else:
                        break
        else:
            results = []
            for c, ct in enumerate(candidate_tuples):
                results.append(evaluate_pairing(ct))
                print("\rCompleted candidate: " + str(c + 1) + "/" + str(n), end="")

        # Extract fitness values.
        player_fit_values = [res[0] for res in results]
        drone_fit_values = [res[1] for res in results]

        # Update the NES optimizers.
        player_nes.tell(player_candidates, player_fit_values)
        drone_nes.tell(drone_candidates, drone_fit_values)

        gen_time = time.time() - start_gen
        pop_size = len(player_candidates)
        print(f"Generation {generation + 1}: Time taken = {gen_time:.2f} sec, Population size = {pop_size}")
        print("  Best player fitness:", -min(player_fit_values))
        print("  Best drone  fitness:", -min(drone_fit_values))

        # Save best models from this generation.
        best_player_candidate, _ = player_nes.result()
        best_drone_candidate, _ = drone_nes.result()
        best_player_model = create_player_hunter_model()
        best_drone_model = create_drone_hunter_model()
        best_player_model.set_weights(unflatten_weights(best_player_candidate, best_player_model.get_weights()))
        best_drone_model.set_weights(unflatten_weights(best_drone_candidate, best_drone_model.get_weights()))
        best_player_model.save(save_path + "player.keras")
        best_drone_model.save(save_path + "drone.keras")

        if player_nes.stop() or drone_nes.stop():
            break

    print("Co-evolution complete!")
