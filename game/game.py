import pygame
import numpy as np
import time
import random
import math
from config import (MAP_HEIGHT, MAP_WIDTH, TILE_SIZE, COLOR_FLOOR,
                    COLOR_WALL, COLOR_EXIT, PLAYER_RADIUS, DRONE_RADIUS, COLOR_START,
                    COLOR_PLAYER, COLOR_DRONE, SCREEN_WIDTH, SCREEN_HEIGHT)
from level import new_level
from entities.player import Player
from entities.drone import batch_update_drones, Drone
from utils import distance
from simulation import simulate_game
import tensorflow as tf


def draw_exit_arrow(screen, player_x, player_y, exit_x, exit_y, screen_width, screen_height):
    """ Draw an arrow at the edge of the screen pointing toward the exit. """
    import math
    ARROW_SIZE = 30

    # Calculate direction to exit
    dx, dy = exit_x - player_x, exit_y - player_y
    angle = math.atan2(dy, dx)

    # Calculate position at screen edge
    max_x = screen_width // 2 - ARROW_SIZE
    max_y = screen_height // 2 - ARROW_SIZE
    scale_factor = min(max_x / abs(dx) if dx else max_x, max_y / abs(dy) if dy else max_y)

    edge_x = screen_width // 2 + dx * scale_factor
    edge_y = screen_height // 2 + dy * scale_factor

    # Keep the arrow inside the screen
    edge_x = max(ARROW_SIZE, min(screen_width - ARROW_SIZE, edge_x))
    edge_y = max(ARROW_SIZE, min(screen_height - ARROW_SIZE, edge_y))

    # Calculate arrow triangle points
    tip = (edge_x, edge_y)
    left = (edge_x - ARROW_SIZE * math.cos(angle - math.pi / 6),
            edge_y - ARROW_SIZE * math.sin(angle - math.pi / 6))
    right = (edge_x - ARROW_SIZE * math.cos(angle + math.pi / 6),
             edge_y - ARROW_SIZE * math.sin(angle + math.pi / 6))

    pygame.draw.polygon(screen, COLOR_EXIT, [tip, left, right])


def run_manual_mode(USE_PLAYER_NN=True, USE_DRONE_NN=True, path="../models_cma/"):
    if path[-1] != '/':
        path += '/'
    player_model_path_hunter = path + "player_hunter.keras"
    drone_model_path_hunter = path + "drone_hunter.keras"
    if USE_PLAYER_NN:
        try:
            model_hunter_player = tf.keras.models.load_model(player_model_path_hunter)
        except:
            from models import create_player_hunter_model
            model_hunter_player = create_player_hunter_model()
    if USE_DRONE_NN:
        try:
            model_hunter_drone = tf.keras.models.load_model(drone_model_path_hunter)
        except:
            from models import create_drone_hunter_model
            model_hunter_drone = create_drone_hunter_model()

    pygame.init()
    infoObject = pygame.display.Info()
    screen_w, screen_h = infoObject.current_w, infoObject.current_h
    #screen_w, screen_h = 600, 600
    screen = pygame.display.set_mode((screen_w, screen_h))
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Drone Escape")
    clock = pygame.time.Clock()

    dungeon, player, drones, exit_rect, player_start = new_level()

    # Camera Offsets
    camera_x, camera_y = player.x - SCREEN_WIDTH // 2, player.y - SCREEN_HEIGHT // 2

    running = True
    step = 0
    import pandas as pd
    while running:
        step += 1
        dt = clock.tick(30) / 1000.0  # Convert to seconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        if USE_PLAYER_NN:
            player.update_nn(dt, dungeon, drones, model_hunter_player)
        else:
            player.update_keyboard(dt, dungeon)

        if USE_DRONE_NN:
            batch_update_drones(drones, dt, dungeon, player, model_hunter_drone)
        else:
            for drone in drones:
                drone.update_random(dt, dungeon, drones)

        """
        if step % 1 == 0:
            player_sensors = player.get_sensors(dungeon, drones)
            player_sensors_dict = {
                f"Angle {player.sensor_angles[i]}": [player_sensors[i], player_sensors[i + len(player.sensor_angles)]]
                for i in range(len(player.sensor_angles))}
            player_sensors_dict[f"Position"] = [player_sensors[-6], player_sensors[-5]]
            player_sensors_dict[f"Velocity"] = [player_sensors[-4], player_sensors[-3]]
            player_sensors_dict[f"Goal"] = [player_sensors[-2], player_sensors[-1]]
            sensors_pandas = pd.DataFrame(player_sensors_dict)
            with pd.option_context('display.max_rows', None, 'display.max_columns',
                                   None):  # more options can be specified also
                print ("PLAYER")
                print(sensors_pandas)
                print()
                print()
            for d_num, drone in enumerate(drones):
                drone_sensors = drone.get_sensors(player, drones, dungeon)
                drone_sensor_dict = {
                    f"Angle {drone.sensor_angles[i]}": [drone_sensors[i], drone_sensors[i + len(drone.sensor_angles)]]
                    for i in range(len(drone.sensor_angles))}
                #drone_sensor_dict[f"Position"] = [drone_sensors[-6], drone_sensors[-5]]
                #drone_sensor_dict[f"Velocity"] = [drone_sensors[-4], drone_sensors[-3]]
                #drone_sensor_dict[f"Goal"] = [drone_sensors[-2], drone_sensors[-1]]
                drone_sensor_pandas = pd.DataFrame(drone_sensor_dict)
                with pd.option_context('display.max_rows', None, 'display.max_columns',
                                        None):
                    print (f"Drone {d_num}")
                    print(drone_sensor_pandas)
                    print ()
                    print ()"""
        # Adjust Camera Offset to keep player centered, but fix to the screen edges if player moves to the edge
        if screen_w < MAP_WIDTH * TILE_SIZE:
            camera_x = player.x - screen_w // 2
            max_camera_x = MAP_WIDTH * TILE_SIZE - screen_w
            if camera_x < 0:
                camera_x = 0
            elif camera_x > max_camera_x:
                camera_x = max_camera_x
            camera_x = max(0, min(camera_x, max_camera_x))
        else:
            camera_x = MAP_WIDTH * TILE_SIZE // 2 - screen_w // 2

        if screen_h < MAP_HEIGHT * TILE_SIZE:
            camera_y = player.y - screen_h // 2
            max_camera_y = MAP_HEIGHT * TILE_SIZE - screen_h
            if camera_y < 0:
                camera_y = 0
            elif camera_y > max_camera_y:
                camera_y = max_camera_y
            camera_y = max(0, min(camera_y, max_camera_y))
        else:
            camera_y = MAP_HEIGHT * TILE_SIZE // 2 - screen_h // 2

        # Check if player reached the exit
        if exit_rect.collidepoint(float(player.x), float(player.y)):
            dungeon, player, drones, exit_rect, player_start = new_level()

        # Check if player is caught
        for drone in drones:
            if distance((player.x, player.y), (drone.x, drone.y)) < (PLAYER_RADIUS + DRONE_RADIUS):
                player.x, player.y = player.spawn
                player.vx, player.vy = 0, 0
                for drone in drones:
                    drone.x, drone.y = drone.spawn
                    drone.vx, drone.vy = 0, 0
                break

        # Draw everything relative to camera position
        screen.fill((0, 0, 0))
        for ty in range(int(camera_y // TILE_SIZE - 1), int(camera_y // TILE_SIZE + screen_h // TILE_SIZE + 1), 1):
            for tx in range(int(camera_x // TILE_SIZE - 1), int(camera_x // TILE_SIZE + screen_w // TILE_SIZE + 1), 1):
                if tx < 0 or tx >= MAP_WIDTH or ty < 0 or ty >= MAP_HEIGHT:
                    color = COLOR_WALL
                else:
                    color = COLOR_FLOOR if dungeon[ty, tx] == 1 else COLOR_WALL
                tile_x = tx * TILE_SIZE - camera_x
                tile_y = ty * TILE_SIZE - camera_y
                pygame.draw.rect(screen, color, (tile_x, tile_y, TILE_SIZE, TILE_SIZE))

        # Draw Exit
        if 0 <= exit_rect.x - camera_x < screen_w and 0 <= exit_rect.y - camera_y < screen_h:
            pygame.draw.rect(screen, COLOR_EXIT,
                             (exit_rect.x - camera_x, exit_rect.y - camera_y, TILE_SIZE, TILE_SIZE))
        else:
            draw_exit_arrow(screen, camera_x + screen_w // 2, camera_y + screen_h // 2,
                            exit_rect.x + TILE_SIZE / 2, exit_rect.y + TILE_SIZE / 2, screen_w - 100, screen_h - 100)
        # Draw Start
        if 0 <= player_start[0] - TILE_SIZE // 2 - camera_x < screen_w and 0 <= player_start[
            1] - TILE_SIZE // 2 - camera_y < screen_h:
            pygame.draw.rect(screen, COLOR_START,
                             (player_start[0] - TILE_SIZE // 2 - camera_x, player_start[1] - TILE_SIZE // 2 - camera_y,
                              TILE_SIZE, TILE_SIZE))

        # Draw Player (always at the center of screen)
        if 0 <= player.x - camera_x < screen_w and 0 <= player.y - camera_y < screen_h:
            pygame.draw.circle(screen, COLOR_PLAYER,
                               (int(player.x - camera_x), int(player.y - camera_y)), PLAYER_RADIUS)

        # Draw Drones
        for drone in drones:
            drone_x = int(drone.x - camera_x)
            drone_y = int(drone.y - camera_y)
            if 0 <= drone_x < screen_w and 0 <= drone_y < screen_h:
                pygame.draw.circle(screen, COLOR_DRONE, (drone_x, drone_y), DRONE_RADIUS)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run_manual_mode(False, False)
