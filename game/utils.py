import math
import numpy as np
import numba
from tensorflow import dynamic_stitch

from config import (TILE_SIZE, MAP_WIDTH, MAP_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT, PLAYER_RADIUS, DRONE_RADIUS,
                    PLAYER_SPEED, DRONE_SPEED)

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
    angles_to_entities = np.pi - np.arctan2((y - entities_y), (x - entities_x))
    distance_to_entities = np.sqrt(np.square(entities_x - x) + np.square(entities_y - y))
    paralax_angle = np.arctan2(entities_radius, distance_to_entities)
    seen_mask = np.abs((angles_to_entities - angle) % (2*math.pi)) < paralax_angle
    if not np.any(seen_mask):
        return None, None
    else:
        return np.min(distance_to_entities[seen_mask]), entities_type[seen_mask][np.argmin(distance_to_entities[seen_mask])]

def sensor_check_exit(x, y, angle, exit_x, exit_y):
    angle_to_exit = np.pi - np.arctan2(exit_y - y, exit_x - x)
    distance_to_exit = np.sqrt((exit_x - x) ** 2 + (exit_y - y) ** 2)
    paralax_angle = np.arctan2(TILE_SIZE, distance_to_exit)
    if np.abs((angle_to_exit - angle) % (2*math.pi)) < paralax_angle:
        return distance_to_exit
    else:
        return None

def get_sensor_at_angle(x, y, angle, dungeon, player_pos, drones_pos, exit_pos, max_distance=SCREEN_WIDTH):
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
            ent_type.append(0.5)
    if player_pos[0] != x and player_pos[1] != y:
        ent_x.append(player_pos[0])
        ent_y.append(player_pos[1])
        ent_rad.append(PLAYER_RADIUS)
        ent_type.append(1)
    dist_e, type_e = sensor_check_entities(x, y, angle_rad, np.array(ent_x), np.array(ent_y), np.array(ent_rad), np.array(ent_type))
    dist_exit, type_exit = sensor_check_exit(x, y, angle_rad, exit_pos[0], exit_pos[1]), -1
    if dist_exit is None:
        dist_exit = max_distance
        type_exit = 0
    if dist_e is None:
        dist_e = max_distance
        type_e = 0
    return min(dist_w, dist_e, dist_exit), [type_w, type_e, type_exit][np.argmin([dist_w, dist_e, dist_exit])]

def move_entity_logic(accel_x, accel_y, vx, vy, x, y, dt, dungeon, entity_speed=PLAYER_SPEED, entity_radius=PLAYER_RADIUS, drones=[]):
    # Drag calculation
    static_drag = entity_speed * 0.1
    static_drag_x = 0
    static_drag_y = 0
    if vx != 0:
        if abs(vx) > static_drag:
            static_drag_x = math.copysign(static_drag, vx)
        else:
            static_drag_x = 0
    if vy != 0:
        if abs(vy) > static_drag:
            static_drag_y = math.copysign(static_drag, vy)
        else:
            static_drag_y = 0
    dynamic_drag_x = 0.01 * vx
    dynamic_drag_y = 0.01 * vy
    vx -= static_drag_x + dynamic_drag_x
    vy -= static_drag_y + dynamic_drag_y
    # Velocity update
    if accel_x or accel_y:
        ACCEL_FACTOR = 10 * entity_speed # adjust as needed
        vx += accel_x * dt * ACCEL_FACTOR - 0.01 * vx
        vy += accel_y * dt * ACCEL_FACTOR - 0.01 * vy
    # Optional: clamp velocity to a maximum speed (entity_speed)
    max_velocity = entity_speed
    current_speed = math.hypot(vx, vy)
    if current_speed > max_velocity:
        vx = (vx / current_speed) * max_velocity
        vy = (vy / current_speed) * max_velocity
    # --- Horizontal Movement ---
    new_x = x + vx * dt
    if collides_with_walls_numba(new_x, y, entity_radius, dungeon):
        vx = -vx  # bounce horizontally
        new_x = x + vx * dt
        if collides_with_walls_numba(new_x, y, entity_radius, dungeon):
            new_x = x
            vx = 0

    # --- Vertical Movement ---
    new_y = y + vy * dt
    if collides_with_walls_numba(x, new_y, entity_radius, dungeon):
        vy = -vy  # bounce vertically
        new_y = y + vy * dt
        if collides_with_walls_numba(x, new_y, entity_radius, dungeon):
            new_y = y
            vy = 0

    # Optionally check for collisions with other drones
    if any(distance((new_x, new_y), (d.x, d.y)) < DRONE_RADIUS * 2 for d in drones):
        vx = -vx
        vy = -vy
        new_x = x + vx * dt
        new_y = y + vy * dt
        if (collides_with_walls_numba(new_x, y, entity_radius, dungeon) or
                collides_with_walls_numba(x, new_y, entity_radius, dungeon) or
                collides_with_walls_numba(new_x, new_y, entity_radius, dungeon)
        ):
            new_x, new_y = x, y
            vx = vy = 0

    # Update position if the new location is valid
    mask = np.kron(np.array(dungeon, dtype=np.int32), np.ones((TILE_SIZE, TILE_SIZE), dtype=np.int32))
    if mask[int(new_y), int(new_x)] == 1:
        x, y = new_x, new_y
    return x, y, vx, vy