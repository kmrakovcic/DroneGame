import os.path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
import pygame
import random
import math
import numpy as np
import tensorflow as tf
import concurrent.futures
import numba
import cma

# === Constants ===
TILE_SIZE = 32
MAP_WIDTH = 30  # in tiles
MAP_HEIGHT = 20  # in tiles
SCREEN_WIDTH = MAP_WIDTH * TILE_SIZE
SCREEN_HEIGHT = MAP_HEIGHT * TILE_SIZE

PLAYER_SPEED = 100.0
DRONE_SPEED = 100.0

PLAYER_RADIUS = 10
DRONE_RADIUS = 10

COLOR_WALL = (100, 100, 100)
COLOR_FLOOR = (200, 200, 200)
COLOR_PLAYER = (50, 200, 50)
COLOR_DRONE = (200, 50, 50)
COLOR_EXIT = (50, 50, 200)

ROOM_MAX_SIZE = 8
ROOM_MIN_SIZE = 4
MAX_ROOMS = 15

BASE_DRONE_WALL_PENALTY = 0.5  # Base penalty per failed drone move
PLAYER_PROGRESS_SCALE = 1.0  # Bonus per unit progress (reduced distance to goal)
DRONE_DISTANCE_SCALE = 0.1  # Penalty per unit average distance from player

# === Mode Selection ===
GAME_MODE = "training"  # "training" or "manual"
USE_DRONE_NN = True

# NEW: Flag to choose parallel or sequential evaluation.
USE_PARALLEL_EVALUATION = True

# === Numba-accelerated Functions ===

@numba.njit
def circle_rect_collision_numba(cx, cy, radius, left, top, right, bottom):
    # Clamp the circle center to the rectangle boundaries.
    closest_x = max(left, min(cx, right))
    closest_y = max(top, min(cy, bottom))
    # Test if the distance from (cx,cy) to the closest point is less than the radius.
    return ((cx - closest_x) ** 2 + (cy - closest_y) ** 2) < (radius ** 2)

# --- New collision function ---
@numba.njit
def collides_with_walls_numba(x, y, radius, dungeon):
    """
    Returns True if a circle at (x,y) with the given radius collides with any wall.
    Walls are defined as dungeon tiles with value 0, or if (x,y) is out of bounds.
    """
    left = x - radius
    right = x + radius
    top = y - radius
    bottom = y + radius
    left_tile = int(left // TILE_SIZE)
    right_tile = int(right // TILE_SIZE)
    top_tile = int(top // TILE_SIZE)
    bottom_tile = int(bottom // TILE_SIZE)
    for ty in range(top_tile, bottom_tile + 1):
        for tx in range(left_tile, right_tile + 1):
            # Out-of-bound tiles count as walls.
            if tx < 0 or tx >= MAP_WIDTH or ty < 0 or ty >= MAP_HEIGHT:
                return True
            if dungeon[ty, tx] == 0:
                t_left = tx * TILE_SIZE
                t_top = ty * TILE_SIZE
                t_right = t_left + TILE_SIZE
                t_bottom = t_top + TILE_SIZE
                if circle_rect_collision_numba(x, y, radius, t_left, t_top, t_right, t_bottom):
                    return True
    return False

# === Basic Helper Functions ===
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def one_hot(index, length):
    vec = [0] * length
    if 0 <= index < length:
        vec[index] = 1
    return vec


# --- Helper Functions to Flatten and Unflatten Weights ---

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

# --- New helper function for paired evaluation ---
def evaluate_pairing(args):
    """
    Given a tuple (p_candidate, d_candidate, num_evals),
    run simulate_game num_evals times and return the average (player_fitness, drone_fitness).
    (Note: fitness functions return negative values so that CMA-ES minimizes.)
    """
    p_candidate, d_candidate, num_evals = args
    player_fits = []
    drone_fits = []
    for _ in range(num_evals):
        pf = fitness_player(p_candidate, d_candidate)
        df = fitness_drone(p_candidate, d_candidate)
        player_fits.append(pf)
        drone_fits.append(df)
    return np.mean(player_fits), np.mean(drone_fits)

# === (Unchanged) Sensor Functions for Drones and Player ===
# [Keep your sensor functions as is...]
# For brevity, these functions are unchanged:
def get_left_sensor(x, y, dungeon, player_pos, drones_pos):
    tile_x = int(x // TILE_SIZE)
    tile_y = int(y // TILE_SIZE)
    row = dungeon[tile_y, :]
    if tile_x > 0:
        indices = np.arange(tile_x)
        wall_candidates = indices[row[indices] == 0]
        wall_dist = x - ((wall_candidates[-1] + 1) * TILE_SIZE) if wall_candidates.size > 0 else x
    else:
        wall_dist = x
    candidate = wall_dist
    sensor_type = 0.0
    if int(player_pos[1] // TILE_SIZE) == tile_y and player_pos[0] < x:
        cand = x - player_pos[0]
        if cand < candidate:
            candidate = cand
            sensor_type = 0.5
    if drones_pos.size > 0:
        mask = ((drones_pos[:, 1] // TILE_SIZE).astype(int) == tile_y) & (drones_pos[:, 0] < x)
        if np.any(mask):
            cand_arr = x - drones_pos[mask, 0]
            min_cand = np.min(cand_arr)
            if min_cand < candidate:
                candidate = min_cand
                sensor_type = 1.0
    return candidate, sensor_type

def get_right_sensor(x, y, dungeon, player_pos, drones_pos):
    tile_x = int(x // TILE_SIZE)
    tile_y = int(y // TILE_SIZE)
    row = dungeon[tile_y, :]
    if tile_x < MAP_WIDTH - 1:
        indices = np.arange(tile_x + 1, MAP_WIDTH)
        wall_candidates = indices[row[indices] == 0]
        wall_dist = (wall_candidates[0] * TILE_SIZE) - x if wall_candidates.size > 0 else SCREEN_WIDTH - x
    else:
        wall_dist = SCREEN_WIDTH - x
    candidate = wall_dist
    sensor_type = 0.0
    if int(player_pos[1] // TILE_SIZE) == tile_y and player_pos[0] > x:
        cand = player_pos[0] - x
        if cand < candidate:
            candidate = cand
            sensor_type = 0.5
    if drones_pos.size > 0:
        mask = ((drones_pos[:, 1] // TILE_SIZE).astype(int) == tile_y) & (drones_pos[:, 0] > x)
        if np.any(mask):
            cand_arr = drones_pos[mask, 0] - x
            min_cand = np.min(cand_arr)
            if min_cand < candidate:
                candidate = min_cand
                sensor_type = 1.0
    return candidate, sensor_type

def get_up_sensor(x, y, dungeon, player_pos, drones_pos):
    tile_x = int(x // TILE_SIZE)
    tile_y = int(y // TILE_SIZE)
    col = dungeon[:, tile_x]
    if tile_y > 0:
        indices = np.arange(tile_y)
        wall_candidates = indices[col[indices] == 0]
        wall_dist = y - ((wall_candidates[-1] + 1) * TILE_SIZE) if wall_candidates.size > 0 else y
    else:
        wall_dist = y
    candidate = wall_dist
    sensor_type = 0.0
    if int(player_pos[0] // TILE_SIZE) == tile_x and player_pos[1] < y:
        cand = y - player_pos[1]
        if cand < candidate:
            candidate = cand
            sensor_type = 0.5
    if drones_pos.size > 0:
        mask = ((drones_pos[:, 0] // TILE_SIZE).astype(int) == tile_x) & (drones_pos[:, 1] < y)
        if np.any(mask):
            cand_arr = y - drones_pos[mask, 1]
            min_cand = np.min(cand_arr)
            if min_cand < candidate:
                candidate = min_cand
                sensor_type = 1.0
    return candidate, sensor_type

def get_down_sensor(x, y, dungeon, player_pos, drones_pos):
    tile_x = int(x // TILE_SIZE)
    tile_y = int(y // TILE_SIZE)
    col = dungeon[:, tile_x]
    if tile_y < MAP_HEIGHT - 1:
        indices = np.arange(tile_y + 1, MAP_HEIGHT)
        wall_candidates = indices[col[indices] == 0]
        wall_dist = (wall_candidates[0] * TILE_SIZE) - y if wall_candidates.size > 0 else SCREEN_HEIGHT - y
    else:
        wall_dist = SCREEN_HEIGHT - y
    candidate = wall_dist
    sensor_type = 0.0
    if int(player_pos[0] // TILE_SIZE) == tile_x and player_pos[1] > y:
        cand = player_pos[1] - y
        if cand < candidate:
            candidate = cand
            sensor_type = 0.5
    if drones_pos.size > 0:
        mask = ((drones_pos[:, 0] // TILE_SIZE).astype(int) == tile_x) & (drones_pos[:, 1] > y)
        if np.any(mask):
            cand_arr = drones_pos[mask, 1] - y
            min_cand = np.min(cand_arr)
            if min_cand < candidate:
                candidate = min_cand
                sensor_type = 1.0
    return candidate, sensor_type

def get_drone_sensor_vector(x, y, dungeon, player_pos, drones_pos):
    left_d, left_t = get_left_sensor(x, y, dungeon, player_pos, drones_pos)
    right_d, right_t = get_right_sensor(x, y, dungeon, player_pos, drones_pos)
    up_d, up_t = get_up_sensor(x, y, dungeon, player_pos, drones_pos)
    down_d, down_t = get_down_sensor(x, y, dungeon, player_pos, drones_pos)
    left_norm = left_d / SCREEN_WIDTH
    right_norm = right_d / SCREEN_WIDTH
    up_norm = up_d / SCREEN_HEIGHT
    down_norm = down_d / SCREEN_HEIGHT
    drone_pos_norm = [x / SCREEN_WIDTH, y / SCREEN_HEIGHT]
    player_pos_norm = [player_pos[0] / SCREEN_WIDTH, player_pos[1] / SCREEN_HEIGHT]
    sensor_vector = np.array([left_norm, right_norm, up_norm, down_norm,
                              left_t, right_t, up_t, down_t,
                              drone_pos_norm[0], drone_pos_norm[1],
                              player_pos_norm[0], player_pos_norm[1]], dtype=np.float32)
    return sensor_vector

def get_player_sensor_vector_vectorized(x, y, dungeon, drones_pos, goal):
    tile_x = int(x // TILE_SIZE)
    tile_y = int(y // TILE_SIZE)
    row = dungeon[tile_y, :]
    if tile_x > 0:
        indices = np.arange(tile_x)
        wall_candidates = indices[row[indices] == 0]
        wall_left = x - ((wall_candidates[-1] + 1) * TILE_SIZE) if wall_candidates.size > 0 else x
    else:
        wall_left = x
    if tile_x < MAP_WIDTH - 1:
        indices = np.arange(tile_x + 1, MAP_WIDTH)
        wall_candidates = indices[row[indices] == 0]
        wall_right = (wall_candidates[0] * TILE_SIZE - x) if wall_candidates.size > 0 else SCREEN_WIDTH - x
    else:
        wall_right = SCREEN_WIDTH - x
    col = dungeon[:, tile_x]
    if tile_y > 0:
        indices = np.arange(tile_y)
        wall_candidates = indices[col[indices] == 0]
        wall_up = y - ((wall_candidates[-1] + 1) * TILE_SIZE) if wall_candidates.size > 0 else y
    else:
        wall_up = y
    if tile_y < MAP_HEIGHT - 1:
        indices = np.arange(tile_y + 1, MAP_HEIGHT)
        wall_candidates = indices[col[indices] == 0]
        wall_down = (wall_candidates[0] * TILE_SIZE - y) if wall_candidates.size > 0 else SCREEN_HEIGHT - y
    else:
        wall_down = SCREEN_HEIGHT - y
    candidate_left = wall_left
    candidate_right = wall_right
    candidate_up = wall_up
    candidate_down = wall_down
    type_left = type_right = type_up = type_down = 0.0
    if drones_pos.size > 0:
        mask_left = ((drones_pos[:, 1] // TILE_SIZE).astype(int) == tile_y) & (drones_pos[:, 0] < x)
        if np.any(mask_left):
            cand = x - drones_pos[mask_left, 0]
            min_cand = np.min(cand)
            if min_cand < candidate_left:
                candidate_left = min_cand
                type_left = 1.0
        mask_right = ((drones_pos[:, 1] // TILE_SIZE).astype(int) == tile_y) & (drones_pos[:, 0] > x)
        if np.any(mask_right):
            cand = drones_pos[mask_right, 0] - x
            min_cand = np.min(cand)
            if min_cand < candidate_right:
                candidate_right = min_cand
                type_right = 1.0
        mask_up = ((drones_pos[:, 0] // TILE_SIZE).astype(int) == tile_x) & (drones_pos[:, 1] < y)
        if np.any(mask_up):
            cand = y - drones_pos[mask_up, 1]
            min_cand = np.min(cand)
            if min_cand < candidate_up:
                candidate_up = min_cand
                type_up = 1.0
        mask_down = ((drones_pos[:, 0] // TILE_SIZE).astype(int) == tile_x) & (drones_pos[:, 1] > y)
        if np.any(mask_down):
            cand = drones_pos[mask_down, 1] - y
            min_cand = np.min(cand)
            if min_cand < candidate_down:
                candidate_down = min_cand
                type_down = 1.0
    left_norm = candidate_left / SCREEN_WIDTH
    right_norm = candidate_right / SCREEN_WIDTH
    up_norm = candidate_up / SCREEN_HEIGHT
    down_norm = candidate_down / SCREEN_HEIGHT
    player_pos_norm = [x / SCREEN_WIDTH, y / SCREEN_HEIGHT]
    goal_norm = [goal[0] / SCREEN_WIDTH, goal[1] / SCREEN_HEIGHT]
    sensor_vector = np.array([left_norm, right_norm, up_norm, down_norm,
                              type_left, type_right, type_up, type_down,
                              player_pos_norm[0], player_pos_norm[1],
                              goal_norm[0], goal_norm[1]], dtype=np.float32)
    return sensor_vector

# === Dungeon Generation Classes and Functions ===
class Room:
    def __init__(self, x, y, w, h):
        self.x1 = x
        self.y1 = y
        self.x2 = x + w
        self.y2 = y + h
        self.center = ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def intersects(self, other):
        return (self.x1 <= other.x2 and self.x2 >= other.x1 and
                self.y1 <= other.y2 and self.y2 >= other.y1)

def generate_dungeon():
    dungeon = [[0 for _ in range(MAP_WIDTH)] for _ in range(MAP_HEIGHT)]
    rooms = []
    for _ in range(MAX_ROOMS):
        w = random.randint(ROOM_MIN_SIZE, ROOM_MAX_SIZE)
        h = random.randint(ROOM_MIN_SIZE, ROOM_MAX_SIZE)
        x = random.randint(0, MAP_WIDTH - w - 1)
        y = random.randint(0, MAP_HEIGHT - h - 1)
        new_room = Room(x, y, w, h)
        if any(new_room.intersects(r) for r in rooms):
            continue
        for i in range(new_room.x1, new_room.x2):
            for j in range(new_room.y1, new_room.y2):
                dungeon[j][i] = 1
        if rooms:
            prev_center = rooms[-1].center
            new_center = new_room.center
            if random.randint(0, 1):
                for x_corr in range(min(prev_center[0], new_center[0]), max(prev_center[0], new_center[0]) + 1):
                    dungeon[prev_center[1]][x_corr] = 1
                for y_corr in range(min(prev_center[1], new_center[1]), max(prev_center[1], new_center[1]) + 1):
                    dungeon[y_corr][new_center[0]] = 1
            else:
                for y_corr in range(min(prev_center[1], new_center[1]), max(prev_center[1], new_center[1]) + 1):
                    dungeon[y_corr][prev_center[0]] = 1
                for x_corr in range(min(prev_center[0], new_center[0]), max(prev_center[0], new_center[0]) + 1):
                    dungeon[new_room.center[1]][x_corr] = 1
        rooms.append(new_room)
    return dungeon, rooms

# === TensorFlow Model Creation Functions (Input dim = 12) ===
def create_drone_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(12,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(2, activation='tanh')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_player_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(12,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(2, activation='tanh')
    ])
    return model

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

# === Game Entities ===
class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.distance_covered = 0

    def update_manual(self, dt, dungeon):
        keys = pygame.key.get_pressed()
        dx = dy = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy -= 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy += 1
        if dx or dy:
            norm = math.hypot(dx, dy)
            dx /= norm; dy /= norm
            # Axis-separated movement:
            new_x = self.x + dx * PLAYER_SPEED * dt
            if collides_with_walls_numba(new_x, self.y, PLAYER_RADIUS, dungeon):
                new_x = self.x
            new_y = self.y + dy * PLAYER_SPEED * dt
            if collides_with_walls_numba(self.x, new_y, PLAYER_RADIUS, dungeon):
                new_y = self.y
            self.distance_covered += math.hypot(new_x - self.x, new_y - self.y)
            self.x, self.y = new_x, new_y

    def update_nn(self, dt, dungeon, drones, goal, model):
        # Get sensor vector and NN output
        sensor_vec = get_player_sensor_vector_vectorized(
            self.x, self.y, dungeon,
            np.array([[d.x, d.y] for d in drones], dtype=np.float32),
            goal
        )
        output = model(sensor_vec.reshape(1, -1)).numpy()[0]
        # Compute velocity from output and scale by PLAYER_SPEED.
        # (Make sure the NN outputs are nonzero; if zero, no movement occurs.)
        norm = math.hypot(output[0], output[1])
        if norm:
            vx = (output[0] / norm) * PLAYER_SPEED
            vy = (output[1] / norm) * PLAYER_SPEED
        else:
            vx = vy = 0

        # --- Horizontal Movement (x-axis) ---
        new_x = self.x + vx * dt
        if collides_with_walls_numba(new_x, self.y, PLAYER_RADIUS, dungeon):
            # Bounce horizontally: reverse the x-velocity.
            vx = -vx
            new_x = self.x + vx * dt
            # If still colliding after the bounce, cancel horizontal movement.
            if collides_with_walls_numba(new_x, self.y, PLAYER_RADIUS, dungeon):
                new_x = self.x
                vx = 0

        # --- Vertical Movement (y-axis) ---
        new_y = self.y + vy * dt
        if collides_with_walls_numba(self.x, new_y, PLAYER_RADIUS, dungeon):
            # Bounce vertically: reverse the y-velocity.
            vy = -vy
            new_y = self.y + vy * dt
            # If still colliding after the bounce, cancel vertical movement.
            if collides_with_walls_numba(self.x, new_y, PLAYER_RADIUS, dungeon):
                new_y = self.y
                vy = 0

        # Update the player's position and covered distance.
        mask = np.kron(np.array(dungeon, dtype=np.int32), np.ones((TILE_SIZE, TILE_SIZE), dtype=np.int32))
        if mask[int(new_y), int(new_x)] == 1:
            self.distance_covered += math.hypot(new_x - self.x, new_y - self.y)
            self.x, self.y = new_x, new_y

    def draw(self, screen):
        pygame.draw.circle(screen, COLOR_PLAYER, (int(self.x), int(self.y)), PLAYER_RADIUS)

class Drone:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        directions = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if dx or dy]
        self.current_dx, self.current_dy = random.choice(directions)
        self.direction_timer = random.uniform(5, 10)
        self.distance_covered = 0

    def update_random(self, dt, dungeon, drones):
        self.direction_timer -= dt
        if self.direction_timer <= 0:
            directions = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if dx or dy]
            self.current_dx, self.current_dy = random.choice(directions)
            self.direction_timer = random.uniform(1.5, 3.0)
        new_x = self.x + self.current_dx * DRONE_SPEED * dt
        new_y = self.y + self.current_dy * DRONE_SPEED * dt
        if collides_with_walls_numba(new_x, new_y, DRONE_RADIUS, dungeon):
            self.direction_timer = 0
            return
        if any(distance((new_x, new_y), (d.x, d.y)) < DRONE_RADIUS * 2 for d in drones if d is not self):
            self.direction_timer = 0
            return
        mask = np.kron(np.array(dungeon, dtype=np.int32), np.ones((TILE_SIZE, TILE_SIZE), dtype=np.int32))
        if mask[int(new_y), int(new_x)] == 1:
            self.distance_covered += math.hypot(new_x - self.x, new_y - self.y)
            self.x, self.y = new_x, new_y

    def update_nn(self, dt, dungeon, player, drones, model, output=None):
        if output is None:
            others = np.array([[d.x, d.y] for d in drones if d is not self], dtype=np.float32)
            sensor_vec = get_drone_sensor_vector(self.x, self.y, dungeon, (player.x, player.y), others)
            output = model(sensor_vec.reshape(1, -1)).numpy()[0]
        norm = math.hypot(output[0], output[1])
        if norm:
            vx = (output[0] / norm) * DRONE_SPEED
            vy = (output[1] / norm) * DRONE_SPEED
        else:
            vx = vy = 0

        # --- Horizontal Movement ---
        new_x = self.x + vx * dt
        if collides_with_walls_numba(new_x, self.y, DRONE_RADIUS, dungeon):
            vx = -vx  # bounce horizontally
            new_x = self.x + vx * dt
            if collides_with_walls_numba(new_x, self.y, DRONE_RADIUS, dungeon):
                new_x = self.x
                vx = 0

        # --- Vertical Movement ---
        new_y = self.y + vy * dt
        if collides_with_walls_numba(self.x, new_y, DRONE_RADIUS, dungeon):
            vy = -vy  # bounce vertically
            new_y = self.y + vy * dt
            if collides_with_walls_numba(self.x, new_y, DRONE_RADIUS, dungeon):
                new_y = self.y
                vy = 0

        # Optionally: check for drone-drone collisions and cancel movement if needed.
        if any(distance((new_x, new_y), (d.x, d.y)) < DRONE_RADIUS * 2 for d in drones if d is not self):
            new_x, new_y = self.x, self.y
            vx = vy = 0

        mask = np.kron(np.array(dungeon, dtype=np.int32), np.ones((TILE_SIZE, TILE_SIZE), dtype=np.int32))
        if mask[int(new_y), int(new_x)] == 1:
            self.distance_covered += math.hypot(new_x - self.x, new_y - self.y)
            self.x, self.y = new_x, new_y

    def draw(self, screen):
        pygame.draw.circle(screen, COLOR_DRONE, (int(self.x), int(self.y)), DRONE_RADIUS)

def batch_update_drones(drones, dt, dungeon, player, models):
    sensor_vectors = np.array([get_drone_sensor_vector(d.x, d.y, dungeon, (player.x, player.y),
                                                       np.array([[o.x, o.y] for o in drones if o is not d],
                                                                dtype=np.float32))
                               for d in drones], dtype=np.float32)
    outputs = models(sensor_vectors).numpy()
    for i, drone in enumerate(drones):
        drone.update_nn(dt, dungeon, player, drones, models, outputs[i])
    return outputs

# === Level Setup ===
def new_level():
    dungeon, rooms = generate_dungeon()
    dungeon_np = np.array(dungeon, dtype=np.int32)
    if rooms:
        player_room = random.choice(rooms)
        start_tile = player_room.center
        player_start = (start_tile[0] * TILE_SIZE + TILE_SIZE / 2, start_tile[1] * TILE_SIZE + TILE_SIZE / 2)
        player = Player(*player_start)
        if len(rooms) > 1:
            possible_exits = [r for r in rooms if r != player_room]
            exit_room = random.choice(possible_exits)
        else:
            exit_room = player_room
        exit_tile = exit_room.center
        exit_rect = pygame.Rect(exit_tile[0] * TILE_SIZE, exit_tile[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    else:
        player_start = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
        player = Player(*player_start)
        exit_rect = pygame.Rect(SCREEN_WIDTH - TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE, TILE_SIZE, TILE_SIZE)
        player_room = None
    drones = []
    num_drones = 3
    for dn in range(num_drones):
        while True:
            tx = random.randint(0, MAP_WIDTH - 1)
            ty = random.randint(0, MAP_HEIGHT - 1)
            if dungeon_np[ty, tx] == 1:
                if player_room is not None:
                    if player_room.x1 <= tx < player_room.x2 and player_room.y1 <= ty < player_room.y2:
                        continue
                tile_center = (tx * TILE_SIZE + TILE_SIZE / 2, ty * TILE_SIZE + TILE_SIZE / 2)
                if distance(tile_center, player_start) > TILE_SIZE and not pygame.Rect(exit_rect).collidepoint(tile_center):
                    if not any(distance(tile_center, (d.x, d.y)) < TILE_SIZE for d in drones):
                        drones.append(Drone(tile_center[0], tile_center[1]))
                        break
    return dungeon_np, player, drones, exit_rect, player_start

# === Simulation Function for Genetic Evaluation ===
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


# --- Fitness Functions for Coevolution ---
def fitness_player(flat_player_weights, flat_drone_weights):
    """Evaluate the player fitness when using these candidate weights.
       (Assume higher is better; we will return negative value for minimization.)"""
    # Create player and drone models.
    p_model = create_player_model()
    d_model = create_drone_model()
    p_template = p_model.get_weights()
    d_template = d_model.get_weights()

    p_model.set_weights(unflatten_weights(flat_player_weights, p_template))
    d_model.set_weights(unflatten_weights(flat_drone_weights, d_template))

    player_fitness, drone_fitness, sim_time = simulate_game(p_model, d_model, dt_sim=0.033, max_time=60.0)
    return -player_fitness  # CMA-ES minimizes


def fitness_drone(flat_player_weights, flat_drone_weights):
    """Evaluate the drone fitness using the given candidate weights.
       (Return negative fitness so that CMA-ES minimizes.)"""
    p_model = create_player_model()
    d_model = create_drone_model()
    p_template = p_model.get_weights()
    d_template = d_model.get_weights()

    p_model.set_weights(unflatten_weights(flat_player_weights, p_template))
    d_model.set_weights(unflatten_weights(flat_drone_weights, d_template))

    player_fitness, drone_fitness, sim_time = simulate_game(p_model, d_model, dt_sim=0.033, max_time=60.0)
    return -drone_fitness  # CMA-ES minimizes

# --- Parallel Evaluation Helper ---
def evaluate_candidate(args):
    index, player_weights, drone_weights, dt_sim, max_time, level = args
    p_model = create_player_model()
    p_model.set_weights(player_weights)
    d_model = create_drone_model()
    d_model.set_weights(drone_weights)
    return simulate_game(p_model, d_model, dt_sim, max_time)

# === Genetic Algorithm Training Mode ===
def run_training_mode_genetic():
    n = 300  # population size
    m = 30   # number of best models to select
    max_generations = 100

    # Initialize two CMA-ES instances.
    initial_player = flatten_weights(create_player_model().get_weights())
    initial_drone = flatten_weights(create_drone_model().get_weights())
    player_sigma = 0.5
    drone_sigma = 0.5
    player_es = cma.CMAEvolutionStrategy(initial_player, player_sigma)
    drone_es  = cma.CMAEvolutionStrategy(initial_drone, drone_sigma)

    # Number of independent evaluations per candidate pair.
    num_evals_per_pair = 3

    for generation in range(max_generations):
        start_gen = time.time()
        # Ask for a batch of candidate solutions from each population.
        player_candidates = player_es.ask()
        drone_candidates  = drone_es.ask()

        # Create a list of candidate tuples.
        # Here, we pair candidate i from the player with candidate i from the drone.
        candidate_tuples = [
            (p_candidate, d_candidate, num_evals_per_pair)
            for p_candidate, d_candidate in zip(player_candidates, drone_candidates)
        ]

        # Evaluate all candidate pairs in parallel.
        if USE_PARALLEL_EVALUATION:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(evaluate_pairing, candidate_tuples))
        else:
            results = [evaluate_pairing(ct) for ct in candidate_tuples]

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
        best_player_model.save(f"best_player_cma_0.keras")
        best_drone_model.save(f"best_drone_cma_0.keras")

        if player_es.stop() or drone_es.stop():
            break

    print("Co-evolution complete!")

# === Manual Mode ===
def run_manual_mode(USE_PLAYER_NN=True, USE_DRONE_NN=True):
    player_model_path = "best_player_cma_0.keras"
    drone_model_path = "best_drone_cma_0.keras"
    if USE_PLAYER_NN:
        if os.path.exists(player_model_path):
            best_player_model = tf.keras.models.load_model(player_model_path)
        else:
            best_player_model = create_player_model()
    if USE_DRONE_NN:
        if os.path.exists(drone_model_path):
            best_drone_model = tf.keras.models.load_model(drone_model_path)
        else:
            best_drone_model = create_drone_model()
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Manual Mode")
    clock = pygame.time.Clock()
    dungeon, player, drones, exit_rect, player_start = new_level()
    running = True
    while running:
        dt = clock.tick(30) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        goal = (exit_rect.x + TILE_SIZE / 2, exit_rect.y + TILE_SIZE / 2)
        if USE_PLAYER_NN:
            player.update_nn(dt, dungeon, drones, goal, best_player_model)
        else:
            player.update_manual(dt, dungeon)
        if USE_DRONE_NN:
            batch_update_drones(drones, dt, dungeon, player, best_drone_model)
        else:
            for drone in drones:
                drone.update_random(dt, dungeon, drones)
        if exit_rect.collidepoint(float(player.x), float(player.y)):
            dungeon, player, drones, exit_rect, player_start = new_level()
        for drone in drones:
            if distance((player.x, player.y), (drone.x, drone.y)) < (PLAYER_RADIUS + DRONE_RADIUS):
                player.x, player.y = player_start
                break
        screen.fill((0, 0, 0))
        for ty in range(MAP_HEIGHT):
            for tx in range(MAP_WIDTH):
                color = COLOR_FLOOR if dungeon[ty, tx] == 1 else COLOR_WALL
                pygame.draw.rect(screen, color, (tx * TILE_SIZE, ty * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        pygame.draw.rect(screen, COLOR_EXIT, exit_rect)
        player.draw(screen)
        for drone in drones:
            drone.draw(screen)
        pygame.display.flip()
    pygame.quit()

# === Main Entry Point ===
def main():
    TRAINING_MODE = False  # Set to True for genetic training; False for manual mode.
    USE_PLAYER_NN = False
    USE_DRONE_NN = True
    if TRAINING_MODE:
        run_training_mode_genetic()
    else:
        run_manual_mode(USE_PLAYER_NN, USE_DRONE_NN)

if __name__ == "__main__":
    main()
