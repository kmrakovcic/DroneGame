import pygame
import math
import random
import numpy as np
from config import DRONE_SPEED, DRONE_RADIUS, SCREEN_WIDTH, SCREEN_HEIGHT, COLOR_DRONE, TILE_SIZE, PLAYER_RADIUS
from utils import collides_with_walls_numba, distance, get_sensor_at_angle, move_entity_logic


class Drone:
    def __init__(self, x, y, sensors_angles=[0, 45, 90, 135, 180, 225, 270, 315]):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.spawn = None
        directions = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if dx or dy]
        self.current_dx, self.current_dy = random.choice(directions)
        self.direction_timer = random.uniform(5, 10)
        self.distance_covered = 0
        self.sensor_angles = sensors_angles
        self.sensors = None

    def get_sensors(self, player, drones, dungeon):
        if self.sensors is not None:
            return self.sensors # return cached sensors
        else:
            others = np.array([[d.x, d.y] for d in drones if d is not self], dtype=np.float32)
            self.sensors = get_drone_sensor_vector(self.x, self.y, self.vx, self.vy, self.sensor_angles, dungeon, (player.x, player.y), others)
            return self.sensors

    def update_random(self, dt, dungeon, drones):
        self.sensors = None
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

    def update_nn(self, dt, dungeon, player, drones, model=None, output=None):
        # Get network output if not provided
        if output is None:
            sensor_vec = self.get_sensors(player, drones, dungeon)
            output = model(sensor_vec.reshape(1, -1)).numpy()[0]

        self.sensors = None
        old_x, old_y = self.x, self.y
        old_vx, old_vy = self.vx, self.vy
        # Interpret network output as acceleration (x and y components)
        accel_x = output[0]
        accel_y = output[1]
        norm_a = math.hypot(accel_x, accel_y)
        accel_x /= norm_a if norm_a else 1
        accel_y /= norm_a if norm_a else 1

        self.x, self.y, self.vx, self.vy = move_entity_logic(accel_x, accel_y, self.vx, self.vy, self.x, self.y, dt,
                                                             dungeon, entity_speed=DRONE_SPEED, entity_radius=DRONE_RADIUS,
                                                             drones=[d for d in drones if d is not self])
        # protect player from spawn camping
        if ((abs(self.x - player.spawn[0]) < 2.5*PLAYER_RADIUS and abs(self.y - player.spawn[1]) < 2.5*PLAYER_RADIUS) and
                (player.x == player.spawn[0] and player.y == player.spawn[1])):
            self.vx = -self.vx
            self.vy = -self.vy
            self.x = old_x + self.vx * dt
            self.y = old_y + self.vy * dt

        self.distance_covered += distance((old_x, old_y), (self.x, self.y))


    def update_manual_acceleration(self, ax, ay, dt, dungeon, player, drones):
        output = np.array([ax, ay], dtype=np.float32)
        self.sensors = None
        self.update_nn(dt, dungeon, player, drones, output=output)

    def draw(self, screen):
        pygame.draw.circle(screen, COLOR_DRONE, (int(self.x), int(self.y)), DRONE_RADIUS)

def batch_update_drones(drones, dt, dungeon, player, models):
    sensor_vectors = np.array([d.get_sensors(player, drones, dungeon)
                               for d in drones], dtype=np.float32)
    outputs = models(sensor_vectors).numpy()
    for i, drone in enumerate(drones):
        drone.update_nn(dt, dungeon, player, drones, models, outputs[i])
    return outputs

def get_drone_sensor_vector(x, y, vx, vy, angles, dungeon, player_pos, drones_pos):
    max_len = math.hypot(SCREEN_WIDTH, SCREEN_HEIGHT)
    sensor_vector = np.empty(len(angles) * 2 + 6, dtype=np.float32)
    for i, angle in enumerate(angles):
        distance, type = get_sensor_at_angle(x, y, angle, dungeon, player_pos, drones_pos)
        distance_norm = distance / max_len
        sensor_vector[i] = distance_norm
        sensor_vector[i + len(angles)] = type
    sensor_vector[-6:] = np.array([x / SCREEN_WIDTH, y / SCREEN_HEIGHT, vx / DRONE_SPEED, vy / DRONE_SPEED,
                                      player_pos[0] / SCREEN_WIDTH, player_pos[1] / SCREEN_HEIGHT], dtype=np.float32)
    return sensor_vector