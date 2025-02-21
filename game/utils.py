import math
import numpy as np
import numba
from config import TILE_SIZE, MAP_WIDTH, MAP_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT, PLAYER_RADIUS, DRONE_RADIUS

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

@numba.njit
def sensor_check_walls(x, y, angle, dungeon, max_distance=SCREEN_WIDTH):
    """
    Raycasting sensor that moves tile by tile instead of small steps.
    Ensures exact tile positions are used to match the original sensor functions.

    Returns:
        - Distance to the nearest obstacle (normalized).
          - Type of obstacle detected (0 = wall, 0.5 = player, 1 = drone).
    """
    dx = math.cos(angle)
    dy = math.sin(angle)

    # Start in current tile
    tile_x = int(x // TILE_SIZE)
    tile_y = int(y // TILE_SIZE)
    #print ("Angle ", math.degrees(angle))
    #print("Start tile ", tile_x, tile_y)

    # Step increments in tile space
    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1

    # Next tile boundary crossing points (in pixel space)
    next_x = (tile_x + (step_x > 0)) * TILE_SIZE
    next_y = (tile_y + (step_y > 0)) * TILE_SIZE

    # Distance to next tile boundary
    t_max_x = (next_x - x) / dx if dx != 0 else float('inf')
    t_max_y = (next_y - y) / dy if dy != 0 else float('inf')

    # How much distance is needed to cross a tile in x or y direction
    t_delta_x = abs(TILE_SIZE / dx) if dx != 0 else float('inf')
    t_delta_y = abs(TILE_SIZE / dy) if dy != 0 else float('inf')

    distance = 0.0

    while distance < max_distance:
        # Move to the next tile
        if t_max_x < t_max_y:
            tile_x += step_x
            distance = t_max_x
            t_max_x += t_delta_x
        else:
            tile_y += step_y
            distance = t_max_y
            t_max_y += t_delta_y

        # Out of bounds check
        if tile_x < 0 or tile_x >= MAP_WIDTH or tile_y < 0 or tile_y >= MAP_HEIGHT:
            return distance, 0  # Wall at boundary

        # Check for walls
        if dungeon[tile_y, tile_x] == 0:
            #print ("End tile ", tile_x, tile_y)
            return distance, 0  # Hit a wall

    return max_distance, 0  # No obstacle found

def sensor_check_entities(x, y, angle, entities_x, entities_y, entities_radius, entities_type):
    """
    
    :param x: 
    :param y: 
    :param entities_x: 
    :param entities_y: 
    :return: 
    """
    angle *= -1
    angles_to_entities = np.arctan2(entities_y - y, entities_x - x)
    distance_to_entities = np.sqrt(np.square(entities_x - x) + np.square(entities_y - y))
    paralax_angle = np.arctan2(entities_radius, distance_to_entities)
    seen_mask = np.abs(angles_to_entities - angle) < paralax_angle
    if not np.any(seen_mask):
        return None, None
    else:
        return np.min(distance_to_entities[seen_mask]), entities_type[np.argmin(distance_to_entities[seen_mask])]

def get_sensor_at_angle(x, y, angle, dungeon, player_pos, drones_pos, max_distance=SCREEN_WIDTH):
    angle_rad = math.radians(angle)
    dist_w, type_w = sensor_check_walls(x, y, angle_rad, dungeon, max_distance)
    ent_x = []
    ent_y = []
    ent_rad = []
    ent_type = []
    for dx, dy in drones_pos:
        if dx != x and dy != y:
            ent_x.append(dx)
            ent_y.append(dy)
            ent_rad.append(DRONE_RADIUS)
            ent_type.append(1)
    if player_pos[0] != x and player_pos[1] != y:
        ent_x.append(player_pos[0])
        ent_y.append(player_pos[1])
        ent_rad.append(PLAYER_RADIUS)
        ent_type.append(0.5)
    dist_e, type_e = sensor_check_entities(x, y, angle_rad, np.array(ent_x), np.array(ent_y), np.array(ent_rad), np.array(ent_type))
    if dist_e is None:
        return dist_w, type_w
    elif dist_w < dist_e:
        return dist_w, type_w
    else:
        return dist_e, type_e