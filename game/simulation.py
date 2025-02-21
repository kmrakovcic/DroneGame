import numpy as np

from config import SCREEN_WIDTH, SCREEN_HEIGHT
from level import new_level
from utils import distance
from config import TILE_SIZE, PLAYER_RADIUS, DRONE_RADIUS
from entities.drone import batch_update_drones

"""
def simulate_game(player_model, drone_model, dt_sim=0.1, max_time=60.0):
    dungeon, player, drones, exit_rect, player_start = new_level()
    goal = (exit_rect.x + TILE_SIZE / 2, exit_rect.y + TILE_SIZE / 2)
    avg_player_drone_distance_start = np.mean([distance((player.x, player.y), (d.x, d.y)) for d in drones])
    player_exit_distance_start = distance((player.x, player.y), goal)
    t = 0.0
    player_reached_exit = False
    player_caught = False
    min_distance = player_exit_distance_start
    while t < max_time:
        player.update_nn(dt_sim, dungeon, drones, goal, player_model)
        batch_update_drones(drones, dt_sim, dungeon, player, drone_model)
        cur_distance = distance((player.x, player.y), goal)
        if cur_distance < min_distance:
            min_distance = cur_distance
        if exit_rect.collidepoint(float(player.x), float(player.y)):
            player_reached_exit = True
            break
        for drone in drones:
            if distance((player.x, player.y), (drone.x, drone.y)) < (PLAYER_RADIUS + DRONE_RADIUS):
                player_caught = True
                break
        if player_caught:
            break
        t += dt_sim
    avg_player_drone_distance_end = np.mean([distance((player.x, player.y), (d.x, d.y)) for d in drones])
    player_exit_distance_end = min_distance
    avg_player_drone_distance = (avg_player_drone_distance_start - avg_player_drone_distance_end) / avg_player_drone_distance_start
    player_exit_distance = (player_exit_distance_start - player_exit_distance_end) / player_exit_distance_start
    drone_fitness, player_fitness = calculate_fitness(player, drones, t, player_reached_exit, player_caught,
                                                      SCREEN_WIDTH, SCREEN_HEIGHT, max_time, avg_player_drone_distance,
                                                      player_exit_distance)
    return player_fitness, drone_fitness, t"""

def simulate_game_step(player, drones, dungeon, exit_rect, goal, player_model, drone_model, dt_sim):
    """
    Performs a single simulation step (one dt_sim).
    Returns updated game state information and termination flags.
    """
    player.update_nn(dt_sim, dungeon, drones, goal, player_model)
    batch_update_drones(drones, dt_sim, dungeon, player, drone_model)

    cur_distance = distance((player.x, player.y), goal)

    # Check if player reaches the exit
    player_reached_exit = exit_rect.collidepoint(float(player.x), float(player.y))

    # Check if a drone catches the player
    player_caught = any(
        distance((player.x, player.y), (drone.x, drone.y)) < (PLAYER_RADIUS + DRONE_RADIUS) for drone in drones
    )

    return cur_distance, player_reached_exit, player_caught

def simulate_game_step_manual(player_input, drones_input, player, drones, dungeon, exit_rect, goal, dt_sim):
    """
    Performs a single simulation step (one dt_sim).
    Returns updated game state information and termination flags.
    """
    player.update_manual_velocity(player_input[0], player_input[1], dt_sim, dungeon)
    for i, drone in enumerate(drones):
        drone.update_manual_acceleration(drones_input[i, 0], drones_input[i, 1], dt_sim, dungeon, player, drones)

    cur_distance = distance((player.x, player.y), goal)

    # Check if player reaches the exit
    player_reached_exit = exit_rect.collidepoint(float(player.x), float(player.y))

    # Check if a drone catches the player
    player_caught = any(
        distance((player.x, player.y), (drone.x, drone.y)) < (PLAYER_RADIUS + DRONE_RADIUS) for drone in drones
    )

    return cur_distance, player_reached_exit, player_caught

def simulate_game(player_model, drone_model, dt_sim=0.1, max_time=60.0):
    """
    Runs a full simulation using `simulate_game_step` until a termination condition is met.
    Returns:
      - player_fitness
      - drone_fitness
      - total simulation time
    """
    dungeon, player, drones, exit_rect, player_start = new_level()
    goal = (exit_rect.x + TILE_SIZE / 2, exit_rect.y + TILE_SIZE / 2)

    avg_player_drone_distance_start = np.mean([distance((player.x, player.y), (d.x, d.y)) for d in drones])
    player_exit_distance_start = distance((player.x, player.y), goal)

    t = 0.0
    player_reached_exit = False
    player_caught = False
    min_distance = player_exit_distance_start

    while t < max_time:
        cur_distance, player_reached_exit, player_caught = simulate_game_step(
            player, drones, dungeon, exit_rect, goal, player_model, drone_model, dt_sim
        )

        if cur_distance < min_distance:
            min_distance = cur_distance

        if player_reached_exit or player_caught:
            break

        t += dt_sim

    avg_player_drone_distance_end = np.mean([distance((player.x, player.y), (d.x, d.y)) for d in drones])
    player_exit_distance_end = min_distance
    avg_player_drone_distance = (avg_player_drone_distance_start - avg_player_drone_distance_end) / avg_player_drone_distance_start
    player_exit_distance = (player_exit_distance_start - player_exit_distance_end) / player_exit_distance_start

    drone_fitness, player_fitness = calculate_fitness(
        player, drones, t, player_reached_exit, player_caught,
        SCREEN_WIDTH, SCREEN_HEIGHT, max_time, avg_player_drone_distance, player_exit_distance
    )

    return player_fitness, drone_fitness, t


def calculate_fitness(player, drones, t, player_reached_exit, player_caught,
                      screen_width, screen_height, max_time, avg_player_drone_distance, player_exit_distance):
    player_fitness = 0
    drone_fitness = 0
    if player_reached_exit:
        player_fitness += 50 + 50 * (max_time - t) / max_time
        drone_fitness += -(100 + (max_time - t))
    if player_caught:
        player_fitness += -(50 + 50 * (max_time - t) / max_time)
        drone_fitness += 50 + 50 * (max_time - t) / max_time
    max_screen_distance = distance((0, 0), (screen_width, screen_height))
    player_fitness += avg_player_drone_distance * 50
    drone_fitness += -avg_player_drone_distance * 50
    player_fitness += -player_exit_distance * 50
    drone_fitness += player_exit_distance * 50
    player_distance_covered = player.distance_covered / max_screen_distance
    drone_distance_covered = np.mean([d.distance_covered for d in drones]) / max_screen_distance
    player_fitness += player_distance_covered * 1
    drone_fitness += drone_distance_covered * 1
    return player_fitness, drone_fitness