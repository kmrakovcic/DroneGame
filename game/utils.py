import math
import numpy as np
import numba
from config import TILE_SIZE, MAP_WIDTH, MAP_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT

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