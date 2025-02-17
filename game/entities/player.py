import pygame
import math
import numpy as np
from config import PLAYER_SPEED, PLAYER_RADIUS, SCREEN_WIDTH, SCREEN_HEIGHT, COLOR_PLAYER, TILE_SIZE, MAP_WIDTH, MAP_HEIGHT
from utils import collides_with_walls_numba, distance


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