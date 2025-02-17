import random
import numpy as np
import time
import concurrent.futures
from models import create_player_model, create_drone_model
from training_helpers import evaluate_candidate
from level import new_level
import os

# === Reproduction Functions (Unchanged) ===
def intermediate_crossover_models(parent1, parent2, create_model_fn, mutation_rate, mutation_strength):
    weights1 = parent1.get_weights()
    weights2 = parent2.get_weights()
    new_weights = []
    for w1, w2 in zip(weights1, weights2):
        r = np.random.rand(*w1.shape)
        new_w = r * w1 + (1 - r) * w2
        mutation_mask = np.random.rand(*new_w.shape) < mutation_rate
        new_w += mutation_mask * np.random.randn(*new_w.shape) * mutation_strength
        new_weights.append(new_w)
    new_model = create_model_fn()
    new_model.set_weights(new_weights)
    return new_model

def discrete_crossover_models(parent1, parent2, create_model_fn, mutation_rate, mutation_strength):
    weights1 = parent1.get_weights()
    weights2 = parent2.get_weights()
    new_weights = []
    for w1, w2 in zip(weights1, weights2):
        r = np.random.rand(*w1.shape) < 0.5
        new_w = r * w1 + (~r) * w2
        mutation_mask = np.random.rand(*new_w.shape) < mutation_rate
        new_w = mutation_mask * np.random.randn(*new_w.shape) + (~mutation_mask) * new_w
        new_weights.append(new_w)
    new_model = create_model_fn()
    new_model.set_weights(new_weights)
    return new_model

def generate_new_population(parents, n, create_model_fn, mutation_rate, mutation_strength):
    new_population = []
    while len(new_population) < n:
        p1 = random.choice(parents)
        p2 = random.choice(parents)
        offspring = discrete_crossover_models(p1, p2, create_model_fn, mutation_rate, mutation_strength)
        new_population.append(offspring)
    return new_population

# === Genetic Algorithm Training Mode ===
def run_training(save_path, use_parallel_evaluation=True):
    n = 300  # population size
    m = 30  # number of best models to select
    num_epochs = 1000
    dt_sim = 0.033  # 30 FPS
    max_time = 60.0
    mutation_rate = 0.1
    mutation_strength = 0.05

    if save_path[-1] != '/':
        save_path += '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    player_population = [create_player_model() for _ in range(n)]
    drone_population = [create_drone_model() for _ in range(n)]

    for epoch in range(num_epochs):
        start_epoch = time.time()
        level = new_level()  # use the same level for all candidate evaluations this epoch
        args_list = []
        for i in range(n):
            args_list.append((i,
                              player_population[i].get_weights(),
                              drone_population[i].get_weights(),
                              dt_sim, max_time, level))
        if use_parallel_evaluation:
            restarts=0
            while True:
                try:
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        results = list(executor.map(evaluate_candidate, args_list, timeout=240))
                    # If successful, break out of the retry loop and proceed
                    break

                except concurrent.futures.TimeoutError:
                    if restarts <= 5:
                        restarts+=1
                        print(f"Timeout occurred in epoch {epoch + 1}! Restarting the epoch...")
                    else:
                        break
        else:
            results = []
            for c, args in enumerate(args_list):
                results.append(evaluate_candidate(args))
                print ("\rCompleted candidate: "+str(c)+"/"+str(n), end="")
        player_fitnesses = [res[0] for res in results]
        drone_fitnesses = [res[1] for res in results]
        sim_times = [res[2] for res in results]
        avg_sim = sum(sim_times) / len(sim_times)
        elapsed = time.time() - start_epoch
        print(f"Epoch {epoch + 1}: Avg in-game time = {avg_sim:.2f} s, epoch elapsed = {elapsed:.2f} s")
        print(f"  Best player fitness: {max(player_fitnesses):.2f}")
        print(f"  Best drone fitness: {max(drone_fitnesses):.2f}")
        best_player_idx = np.argsort(player_fitnesses)[-m:]
        best_drone_idx = np.argsort(drone_fitnesses)[-m:]
        best_player_models = [player_population[i] for i in best_player_idx]
        best_drone_models = [drone_population[i] for i in best_drone_idx]
        if (epoch % 10 == 0) or (epoch == num_epochs - 1):
            for i, model in enumerate(best_player_models):
                model.save(save_path+f"best_player_{i}.keras")
            for i, model in enumerate(best_drone_models):
                model.save(save_path+f"best_drone_{i}.keras")
            best_drone_models[0].save(save_path+"drone.keras")
            best_player_models[0].save(save_path+"player.keras")
        else:
            best_drone_models[0].save(save_path+"drone.keras")
            best_player_models[0].save(save_path+"player.keras")
        # Generate new population with dynamic mutation rates
        player_population = generate_new_population(best_player_models, n, create_player_model, mutation_rate, mutation_strength)
        drone_population = generate_new_population(best_drone_models, n, create_drone_model, mutation_rate, mutation_strength)
    print("Genetic training complete.")