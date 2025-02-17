import pygame
import math
import random
import numpy as np
from config import DRONE_SPEED, DRONE_RADIUS, SCREEN_WIDTH, SCREEN_HEIGHT, COLOR_DRONE, TILE_SIZE
from utils import collides_with_walls_numba, distance, get_left_sensor, get_right_sensor, get_up_sensor, get_down_sensor


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

