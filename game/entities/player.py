import pygame
import math
import numpy as np
from config import PLAYER_SPEED, PLAYER_RADIUS, SCREEN_WIDTH, SCREEN_HEIGHT, COLOR_PLAYER, TILE_SIZE, MAP_WIDTH, MAP_HEIGHT
from utils import collides_with_walls_numba, distance, get_sensor_at_angle, move_entity_logic


# === Game Entities ===
class Player:
    def __init__(self, x, y, sensors_angles=[0, 45, 90, 135, 180, 225, 270, 315]):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.spawn = None
        self.level_exit = None
        self.distance_covered = 0
        self.sensor_angles = sensors_angles
        self.sensors = None

    def get_sensors(self, dungeon, drones):
        if self.sensors is not None:
            return self.sensors
        else:
            self.sensors = get_player_sensor_vector(
                self.x, self.y, self.vx, self.vy, self.sensor_angles, dungeon,
                np.array([[d.x, d.y] for d in drones], dtype=np.float32),
                self.level_exit)
            return self.sensors

    def update_keyboard(self, dt, dungeon):
        self.sensors = None
        keys = pygame.key.get_pressed()
        accel_x = accel_y = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            accel_x -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            accel_x += 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            accel_y -= 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            accel_y += 1
        self.x, self.y, self.vx, self.vy = move_entity_logic(accel_x, accel_y, self.vx, self.vy, self.x, self.y, dt, dungeon,
                                                             entity_speed=PLAYER_SPEED, entity_radius=PLAYER_RADIUS, drones=[])


    def update_nn(self, dt, dungeon, drones, model):
        # Get sensor vector and NN output
        sensor_vec = self.get_sensors(dungeon, drones)
        output = model(sensor_vec.reshape(1, -1)).numpy()[0]
        self.sensors = None
        accel_x = output[0]
        accel_y = output[1]
        norm_a = math.hypot(accel_x, accel_y)
        accel_x /= norm_a if norm_a else 1
        accel_y /= norm_a if norm_a else 1

        self.x, self.y, self.vx, self.vy = move_entity_logic(accel_x, accel_y, self.vx, self.vy, self.x, self.y, dt,
                                                             dungeon, entity_speed=PLAYER_SPEED, entity_radius=PLAYER_RADIUS, drones=[])

    def update_manual_velocity(self, accel_x, accel_y, dt, dungeon):
        self.sensors = None
        norm_a = math.hypot(accel_x, accel_y)
        accel_x /= norm_a if norm_a else 1
        accel_y /= norm_a if norm_a else 1
        self.x, self.y, self.vx, self.vy = move_entity_logic(accel_x, accel_y, self.vx, self.vy, self.x, self.y, dt,
                                                             dungeon, entity_speed=PLAYER_SPEED,
                                                             entity_radius=PLAYER_RADIUS, drones=[])

    def draw(self, screen):
        pygame.draw.circle(screen, COLOR_PLAYER, (int(self.x), int(self.y)), PLAYER_RADIUS)

def get_player_sensor_vector(x, y, vx, vy, angles, dungeon, drones_pos, goal):
    max_len = math.hypot(SCREEN_WIDTH, SCREEN_HEIGHT)
    sensor_vector = np.empty(len(angles) * 2 + 6, dtype=np.float32)
    for i, angle in enumerate(angles):
        distance, type = get_sensor_at_angle(x, y, angle, dungeon, (x, y), drones_pos, goal)
        distance_norm = distance / max_len
        sensor_vector[i] = distance_norm
        sensor_vector[i + len(angles)] = type
    sensor_vector[-6:] = np.array([x / SCREEN_WIDTH, y / SCREEN_HEIGHT, vx / PLAYER_SPEED, vy / PLAYER_SPEED,
                                      goal[0] / SCREEN_WIDTH, goal[1] / SCREEN_HEIGHT], dtype=np.float32)
    return sensor_vector