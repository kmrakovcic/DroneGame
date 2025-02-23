import math
import numpy as np
import random
import tensorflow as tf
import concurrent.futures
import argparse

from tensorflow.python.keras.utils.version_utils import callbacks

from simulation import simulate_game_step_manual
from level import new_level
from models import create_player_model, create_drone_model

# --- Assume your constants, tile sizes, and sensor functions are defined elsewhere ---
TILE_SIZE = 32
MAP_WIDTH = 30
MAP_HEIGHT = 20
SCREEN_WIDTH = MAP_WIDTH * TILE_SIZE
SCREEN_HEIGHT = MAP_HEIGHT * TILE_SIZE
PLAYER_SPEED = 100.0
DRONE_SPEED = 100.0
PLAYER_RADIUS = 10
DRONE_RADIUS = 10


# For example, assume these functions exist (they are your original implementations):
# from simulation import new_level, generate_dungeon
# from models import create_player_model, create_drone_model
# from sensors import get_player_sensor_vector_vectorized, get_drone_sensor_vector

# For this example, we assume new_level() returns (dungeon, player, drones, exit_rect, player_start)
# and generate_dungeon() returns a dungeon grid and list of rooms.
# We also assume that get_player_sensor_vector_vectorized(x, y, dungeon, drones_pos, goal)
# and get_drone_sensor_vector(x, y, dungeon, player_pos, drones_pos) are defined.

# --- Simple A* Implementation ---
def astar(grid, start, goal):
    """
    A* pathfinding on a 2D grid.
    grid: 2D numpy array (1 for free, 0 for wall)
    start: (row, col)
    goal: (row, col)
    Returns a list of (row, col) from start to goal or None if no path.
    """
    rows, cols = grid.shape
    open_set = {start}
    came_from = {}
    g_score = {start: 0}

    # Using Euclidean distance as heuristic
    def h(cell):
        return math.hypot(cell[0] - goal[0], cell[1] - goal[1])

    f_score = {start: h(start)}

    while open_set:
        current = min(open_set, key=lambda c: f_score.get(c, float('inf')))
        if current == goal:
            # Reconstruct path:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        open_set.remove(current)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-connected grid
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0], neighbor[1]] == 1:
                tentative_g = g_score[current] + 1  # assume cost 1 for each move
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + h(neighbor)
                    open_set.add(neighbor)
    return None  # no path found

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

    player_cell = (int(player.y // TILE_SIZE), int(player.x // TILE_SIZE))
    path_player = astar(grid, player_cell, exit_cell)
    pixel_path_player = get_pixel_path(path_player, TILE_SIZE)
    while (t < max_time) and (not player_reached_exit) and (not player_caught):
        # Collect sensor data before update
        drone_sensors = [drone.get_sensors(player, drones, dungeon) for drone in drones]
        player_sensors = player.get_sensors(dungeon, drones, goal)

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

        player_outputs.append(player_input)
        player_inputs.append(player_sensors)
        t += dt
    return np.array(player_inputs, dtype=np.float16), np.array(player_outputs, dtype=np.float16), \
        np.array(drone_inputs, dtype=np.float16), np.array(drone_outputs, dtype=np.float16)

def generate_training_data(num_episodes=50, dt=0.033, max_time=60.0, parallel=True):
    """Runs multiple simulation episodes in parallel to collect training data."""
    player_inputs = []
    player_outputs = []
    drone_inputs = []
    drone_outputs = []

    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(generate_episode, [dt] * num_episodes, [max_time] * num_episodes))

        for p_in, p_out, d_in, d_out in results:
            player_inputs.append(p_in)
            player_outputs.append(p_out)
            drone_inputs.append(d_in)
            drone_outputs.append(d_out)
        return (np.concatenate(player_inputs), np.concatenate(player_outputs)), (np.concatenate(drone_inputs), np.concatenate(drone_outputs))
    else:
        for episode in range(num_episodes):
            p_in, p_out, d_in, d_out = generate_episode(dt, max_time)
            player_inputs.append(p_in)
            player_outputs.append(p_out)
            drone_inputs.append(d_in)
            drone_outputs.append(d_out)
        return (np.concatenate(player_inputs), np.concatenate(player_outputs)), (np.concatenate(drone_inputs), np.concatenate(drone_outputs))


# --- Pretrain the Models ---
def train_pretrained_models(num_episodes=50, epochs=10, batch_size=1024, dt=0.033, max_time=60.0):
    # Generate training examples.
    (player_x, player_y), (drone_x, drone_y) = generate_training_data(num_episodes, dt, max_time)

    # Create models (assume these functions are defined in your models module).
    player_model = create_player_model(player_x.shape[1])
    drone_model = create_drone_model(drone_x.shape[1])

    player_model.compile(optimizer='adam', loss='mse')
    drone_model.compile(optimizer='adam', loss='mse')
    reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True)

    print("Pretraining player model...")
    player_model.fit(player_x, player_y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.2,
                     callbacks=[reduce_on_plateau, early_stopping])
    print("Pretraining drone model...")
    drone_model.fit(drone_x, drone_y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True, validation_split=0.2,
                    callbacks=[reduce_on_plateau, early_stopping])

    return player_model, drone_model

def plot_paths_dungeon(dungeon, player, drones, exit_rect):
    exit_center = (exit_rect.x + TILE_SIZE / 2, exit_rect.y + TILE_SIZE / 2)
    exit_cell = (int(exit_center[1] // TILE_SIZE), int(exit_center[0] // TILE_SIZE))
    grid = np.array(dungeon)
    dungeon_mask = np.kron(dungeon, np.ones((TILE_SIZE, TILE_SIZE)))
    player_cell = (int(player.y // TILE_SIZE), int(player.x // TILE_SIZE))
    path_player = astar(grid, player_cell, exit_cell)
    pixel_path_player = get_pixel_path(path_player, TILE_SIZE)
    print (pixel_path_player)
    # put exit on a dungeon mask
    dungeon_mask[int(exit_center[1]) - TILE_SIZE:int(exit_center[1]) + TILE_SIZE,
                 int(exit_center[0]) - TILE_SIZE:int(exit_center[0]) + TILE_SIZE] = 3
    for x, y in pixel_path_player:
        dungeon_mask[y, x] = 2
    plt.imshow(dungeon_mask)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run game or training modes")
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_path', type=str, default="../models_cma/")
    args = parser.parse_args()

    if args.save_path[-1] != '/':
        args.save_path += '/'

    model_player, model_drone = train_pretrained_models(args.episodes, args.epochs)
    model_player.save(args.save_path+"player.keras")
    model_drone.save(args.save_path+"drone.keras")
# --- Example Usage ---
if __name__ == "__main__":
    main()






