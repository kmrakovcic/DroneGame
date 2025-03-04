import math
import os
import time

import numpy as np
import random
import tensorflow as tf
import concurrent.futures
import argparse
import heapq

from tensorflow.python.keras.utils.version_utils import callbacks
from collections import deque

from config import DRONE_NUMBER
from simulation import simulate_game_step_manual
from level import new_level
from models import create_player_hunter_model, create_drone_hunter_model
from config import TILE_SIZE, MAP_WIDTH, MAP_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT, PLAYER_SPEED, DRONE_SPEED, \
    PLAYER_RADIUS, DRONE_RADIUS


def astar(grid, start, goal):
    """
    A* pathfinding with corner clipping prevention.
    grid: 2D numpy array (1 for free, 0 for wall)
    start: (row, col)
    goal: (row, col)
    Returns a list of (row, col) from start to goal or None if no path.
    """
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    # Using Euclidean distance as heuristic
    def h(cell):
        return math.hypot(cell[0] - goal[0], cell[1] - goal[1])

    f_score = {start: h(start)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        neighbors = [
            (current[0] + dx, current[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal directions
                           (-1, -1), (1, -1), (-1, 1), (1, 1)]  # Diagonal directions
        ]

        for neighbor in neighbors:
            r, c = neighbor

            # Check boundaries
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            if grid[r, c] == 0:  # Wall detected
                continue

            # Prevent diagonal corner clipping
            if abs(r - current[0]) == 1 and abs(c - current[1]) == 1:
                if grid[current[0], c] == 0 or grid[r, current[1]] == 0:
                    continue  # Block diagonal move if a wall is in the way

            tentative_g = g_score[current] + math.hypot(r - current[0], c - current[1])

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + h(neighbor)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found



def interpolate_line(start, end):
    """Bresenham's line algorithm to generate all pixels between two points."""
    x1, y1 = start
    x2, y2 = end
    pixels = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while (x1, y1) != (x2, y2):
        pixels.append((x1, y1))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    pixels.append((x2, y2))  # Include endpoint
    return pixels


def get_pixel_path(path, tile_size):
    """Convert A* tile path to a list of all pixels along the route."""
    pixel_path = []

    # Convert tile positions to pixel centers
    waypoints = [(col * tile_size + tile_size // 2, row * tile_size + tile_size // 2) for row, col in path]

    for i in range(len(waypoints) - 1):
        pixel_path.extend(interpolate_line(waypoints[i], waypoints[i + 1]))

    return pixel_path


def find_best_forward_waypoint(player_pos, pixel_path, current_step, max_distance):
    """
    Finds the best forward waypoint that the player can reach without backtracking.

    Args:
        player_pos (tuple): (x, y) position of the player.
        pixel_path (list): List of waypoints [(x, y), ...].
        current_step (int): The index of the current waypoint.
        max_distance (float): Maximum distance the player can travel in one step.

    Returns:
        (int, tuple): The index of the selected waypoint and its (x, y) coordinates.
    """
    best_index = current_step  # Start from current position
    best_waypoint = pixel_path[current_step]

    for i in range(current_step, len(pixel_path)):  # Only scan forward
        waypoint = pixel_path[i]
        distance_to_waypoint = math.hypot(waypoint[0] - player_pos[0], waypoint[1] - player_pos[1])

        if distance_to_waypoint <= max_distance:  # If it's within reach, update target
            best_index = i
            best_waypoint = waypoint
        else:
            break  # Stop once we pass the reachable limit

    return best_index, best_waypoint


def generate_episode(dt, max_time):
    player_inputs = []
    player_outputs = []
    drone_inputs = []
    drone_outputs = []
    dungeon, player, drones, exit_rect, _ = new_level()
    exit_center = (exit_rect.x + TILE_SIZE / 2, exit_rect.y + TILE_SIZE / 2)
    goal = (exit_rect.x + TILE_SIZE / 2, exit_rect.y + TILE_SIZE / 2)
    exit_cell = (int(exit_center[1] // TILE_SIZE), int(exit_center[0] // TILE_SIZE))
    grid = np.array(dungeon)
    t = 0.0
    player_reached_exit = False
    player_caught = False
    step = 0
    drone_steps = [0] * len(drones)
    drones_positions = [[[drone.spawn[0], drone.spawn[1]]] for drone in drones]
    player_positions = [[player.spawn[0], player.spawn[1]]]

    player_cell = (int(player.y // TILE_SIZE), int(player.x // TILE_SIZE))
    path_player = astar(grid, player_cell, exit_cell)
    pixel_path_player = get_pixel_path(path_player, TILE_SIZE)
    while (t < max_time) and (not player_reached_exit) and (not player_caught):
        # Collect sensor data before update
        drone_sensors = [drone.get_sensors(player, drones, dungeon) for drone in drones]
        player_sensors = player.get_sensors(dungeon, drones)

        # === PLAYER PATH FOLLOWING TO EXIT ===
        max_distance = PLAYER_SPEED * dt
        step, target_player_pixel = find_best_forward_waypoint((player.x, player.y), pixel_path_player, step,
                                                               max_distance)

        d_vx = (target_player_pixel[0] - player.x) / dt
        d_vy = (target_player_pixel[1] - player.y) / dt
        ax = (d_vx - player.vx) / dt  # Compute acceleration (a = dv/dt)
        ay = (d_vy - player.vy) / dt
        norm_d = math.hypot(ax, ay)  # * DRONE_SPEED
        ax = ax / norm_d if norm_d > 0 else 0
        ay = ay / norm_d if norm_d > 0 else 0
        norm = math.hypot(ax, ay)
        if norm > 0:
            player_input = np.array([ax / norm, ay / norm], dtype=np.float32)
        else:
            player_input = np.array([0, 0], dtype=np.float32)

        # === DRONE PATH FOLLOWING TO PLAYER'S EXACT POSITION ===
        drone_input = []
        for i, drone in enumerate(drones):
            drone_cell = (int(drone.y // TILE_SIZE), int(drone.x // TILE_SIZE))
            player_cell = (int(player.y // TILE_SIZE), int(player.x // TILE_SIZE))

            # Compute A* path for each drone towards the **player's tile**
            path_drone = astar(grid, drone_cell, player_cell)
            pixel_path_drone = get_pixel_path(path_drone, TILE_SIZE)

            if len(pixel_path_drone) > 0:
                # Find the best reachable waypoint within dt
                drone_steps[i], target_drone_pixel = find_best_forward_waypoint(
                    (drone.x, drone.y), pixel_path_drone, drone_steps[i], DRONE_SPEED * dt
                )

                # Instead of stopping at the grid center, move towards **player's exact position**
                final_target_x, final_target_y = player.x, player.y
                if drone_steps[i] >= len(pixel_path_drone) - 1:
                    target_drone_pixel = (final_target_x, final_target_y)  # Adjust to exact position

                # Compute acceleration (a_x, a_y) for the drone
                d_vx = (target_drone_pixel[0] - drone.x) / dt
                d_vy = (target_drone_pixel[1] - drone.y) / dt
                ax = (d_vx - drone.vx) / dt  # Compute acceleration (a = dv/dt)
                ay = (d_vy - drone.vy) / dt
                norm_d = math.hypot(ax, ay)  # * DRONE_SPEED
                ax = ax / norm_d if norm_d > 0 else 0
                ay = ay / norm_d if norm_d > 0 else 0

                drone_input.append([ax, ay])
            else:
                drone_input.append([0, 0])  # No movement if no valid path
            drone_inputs.append(drone_sensors[i])
            drone_outputs.append(drone_input[i])
        drone_input = np.array(drone_input, dtype=np.float32)
        # Run one step of the simulation
        cur_distance, player_reached_exit, player_caught = simulate_game_step_manual(
            player_input, drone_input, player, drones, dungeon, exit_rect, goal, dt)
        for i in range(len(drones)):
            drones_positions[i].append([drones[i].x, drones[i].y])
        player_positions.append([player.x, player.y])

        player_outputs.append(player_input)
        player_inputs.append(player_sensors)
        t += dt
    return (np.array(player_inputs, dtype=np.float32), np.array(player_outputs, dtype=np.float32), np.array(drone_inputs, dtype=np.float32), np.array(drone_outputs, dtype=np.float32), (dungeon, exit_rect, [player.spawn[0], player.spawn[1]], [[drone.spawn[0], drone.spawn[1]] for drone in drones]), drones_positions, player_positions)


def generate_training_data(num_episodes=50, dt=0.033, max_time=60.0, parallel=True):
    """Runs multiple simulation episodes in parallel to collect training data."""

    player_inputs = np.empty(num_episodes, dtype=object)
    player_outputs = np.empty(num_episodes, dtype=object)
    player_position = np.empty(num_episodes, dtype=object)
    drone_inputs = np.empty(num_episodes, dtype=object)
    drone_outputs = np.empty(num_episodes, dtype=object)
    drone_position = np.empty(num_episodes, dtype=object)
    dungeons = np.empty(num_episodes, dtype=object)

    timeout_per_episode = 0.035  # Timeout per episode in seconds
    if parallel and num_episodes == 1:
        parallel = False  # Force parallel execution for a single episode
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(generate_episode, dt, max_time): i + 1 for i in range(num_episodes)}
            completed = 0
            try:
                for future in concurrent.futures.as_completed(futures, timeout=num_episodes * timeout_per_episode):
                    episode_id = futures[future]
                    try:
                        p_in, p_out, d_in, d_out, (dungeon, exit_rect, player_spawn, drones_spawn), d_pos, p_pos = future.result(timeout=timeout_per_episode)
                        dungeon = np.kron(np.array(dungeon, dtype=np.int32), np.ones((TILE_SIZE, TILE_SIZE), dtype=np.int32))
                        player_inputs[completed] = p_in
                        player_outputs[completed] = p_out
                        player_position[completed] = p_pos
                        drone_inputs[completed] = d_in
                        drone_outputs[completed] = d_out
                        drone_position[completed] = d_pos
                        dungeons[completed] = dungeon
                        completed += 1
                        print(f"\rEpisode: {completed} / {num_episodes} generated", end="")

                    except concurrent.futures.TimeoutError:
                        print(f"\nTimeout occurred in episode {episode_id}, skipping...")

                    except Exception as e:
                        print(f"\nError in episode {episode_id}: {e}")
            except concurrent.futures.TimeoutError:
                print(f"\nTimeout occurred after {completed} episodes, stopping...")
                executor.shutdown(wait=False, cancel_futures=True)
        player_inputs = player_inputs[:completed ]
        player_outputs = player_outputs[:completed]
        player_position = player_position[:completed ]
        drone_inputs = drone_inputs[:completed]
        drone_outputs = drone_outputs[:completed]
        drone_position = drone_position[:completed]
        dungeons = dungeons[:completed]
        print(f"\n{completed} episodes completed")

    else:
        for episode in range(1, num_episodes + 1):
            try:
                p_in, p_out, d_in, d_out, (dungeon, exit_rect, player_spawn, drones_spawn), d_pos, p_pos = generate_episode(dt, max_time)
                dungeon = np.kron(np.array(dungeon, dtype=np.int32), np.ones((TILE_SIZE, TILE_SIZE), dtype=np.int32))
                player_inputs[episode - 1] = p_in
                player_outputs[episode - 1] = p_out
                player_position[episode - 1] = p_pos
                drone_inputs[episode - 1] = d_in
                drone_outputs[episode - 1] = d_out
                drone_position[episode - 1] = d_pos
                dungeons[episode - 1] = dungeon
            except Exception as e:
                print(f"\nError in episode {episode}: {e}")

            print(f"\rEpisode: {episode} / {num_episodes} generated", end="")
    print()
    player_inputs = np.concatenate(player_inputs)
    player_outputs = np.concatenate(player_outputs)
    drone_inputs = np.concatenate(drone_inputs)
    drone_outputs = np.concatenate(drone_outputs)
    return (player_inputs, player_outputs), (drone_inputs, drone_outputs), (player_position, drone_position, dungeons)


# --- Pretrain the Models ---
def train_pretrained_models(player_x, player_y, drone_x, drone_y, epochs=10, batch_size=1024, drone_path=None,
                            player_path=None, template_path=None):
    balance_drones = False
    if balance_drones:
        drone_x_disassembled_masks = [np.any(drone_x[:, 8:16] == i, axis=-1) for i in np.unique(drone_x[:, 8:16])]
        drone_x_disassembled_min = np.min([np.sum(mask) for mask in drone_x_disassembled_masks])
        drone_x_disassembled_indices = [np.random.choice(np.where(mask)[0], drone_x_disassembled_min) for mask in drone_x_disassembled_masks]
        drone_x = np.concatenate([drone_x[indices] for indices in drone_x_disassembled_indices])
        drone_y = np.concatenate([drone_y[indices] for indices in drone_x_disassembled_indices])
    # Create models (assume these functions are defined in your models module).
    if template_path is not None:
        if template_path[-1] != '/':
            template_path += '/'
        try:
            player_model = tf.keras.models.load_model(template_path+"player_hunter_template.keras")
            drone_model = tf.keras.models.load_model(template_path+"drone_hunter_template.keras")
            fine_tune=True
        except:
            player_model = create_player_hunter_model(player_x.shape[1])
            drone_model = create_drone_hunter_model(drone_x.shape[1])
            fine_tune=False
    else:
        player_model = create_player_hunter_model(player_x.shape[1])
        drone_model = create_drone_hunter_model(drone_x.shape[1])
        fine_tune=False

    player_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001 if fine_tune else 0.01), loss='mse')
    drone_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001 if fine_tune else 0.01), loss='mse')
    reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True,
                                                      start_from_epoch=epochs // 2)
    callbacks = [reduce_on_plateau, early_stopping]
    if drone_path is not None:
        checkpoints = tf.keras.callbacks.ModelCheckpoint(drone_path, monitor='val_loss', save_best_only=True)
        callbacks.append(checkpoints)

    print("Pretraining drone model...")
    drone_model.fit(drone_x, drone_y, epochs=epochs, batch_size=batch_size * DRONE_NUMBER, verbose=2, shuffle=True,
                    validation_split=0.2,
                    callbacks=callbacks)
    if drone_path is not None:
        drone_model.save(drone_path)
    print("Pretraining player model...")
    callbacks = [reduce_on_plateau, early_stopping]
    if player_path is not None:
        checkpoints = tf.keras.callbacks.ModelCheckpoint(player_path, monitor='val_loss', save_best_only=True)
        callbacks.append(checkpoints)
    player_model.fit(player_x, player_y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True,
                     validation_split=0.2,
                     callbacks=callbacks)
    if player_path is not None:
        player_model.save(player_path)
    return player_model, drone_model


def plot_paths_dungeon(dungeon, player_pos, drones_pos):
    import matplotlib.pyplot as plt
    drones_pos = np.array(drones_pos)
    player_pos = np.array(player_pos)
    fig, ax = plt.subplots(1)
    ax.imshow(dungeon, cmap="gray")
    ax.plot(player_pos[:, 0], player_pos[:, 1], 'g')
    for d in range(drones_pos.shape[0]):
        ax.plot(drones_pos[d, :, 0], drones_pos[d, :, 1], 'r')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run game or training modes")
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_path', type=str, default="../models_cma/")
    parser.add_argument('--train_data', type=str, default="../DATA/training.npz")
    parser.add_argument('--template_path', type=str, default="")
    args = parser.parse_args()
    if args.save_path[-1] != '/':
        args.save_path += '/'
    if os.path.exists(args.train_data):
        player_x, player_y, drone_x, drone_y = np.load(args.train_data, allow_pickle=True).values()
    else:
        (player_x, player_y), (drone_x, drone_y), _ = generate_training_data(args.episodes, parallel=True)
        os.makedirs(os.path.dirname(args.train_data), exist_ok=True)
        np.savez(args.train_data, player_x=player_x, player_y=player_y, drone_x=drone_x, drone_y=drone_y)
    if not os.exists(args.template_path):
        args.template_path = None
    model_player, model_drone = train_pretrained_models(player_x, player_y, drone_x, drone_y, args.epochs,
                                                        drone_path=args.save_path + "drone_hunter.keras",
                                                        player_path=args.save_path + "player_hunter.keras",
                                                        template_path=args.template_path)
    return model_drone, model_player


# --- Example Usage ---
if __name__ == "__main__":
    #player_x, player_y, drone_x, drone_y = np.load("../DATA/training.npz", allow_pickle=True).values()
    #print ((drone_x[:, 8:16]==0.75).sum())

    main()
    #__, __, (player_pos, drone_pos, dungeons) = generate_training_data(num_episodes=1, parallel=True)
    #plot_paths_dungeon(dungeons[0], player_pos[0], drone_pos[0])
