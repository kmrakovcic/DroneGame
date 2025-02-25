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


def run_manual_mode1(USE_PLAYER_NN=True, USE_DRONE_NN=True, path="../models_ga/"):
    from config import SCREEN_WIDTH, SCREEN_HEIGHT, TILE_SIZE
    from entities.player import Player
    from entities.drone import Drone, batch_update_drones
    import tensorflow as tf

    if path[-1] != '/':
        path += '/'
    player_model_path = path + "player.keras"
    drone_model_path = path + "drone.keras"
    if USE_PLAYER_NN:
        try:
            best_player_model = tf.keras.models.load_model(player_model_path)
        except:
            from models import create_player_model
            best_player_model = create_player_model()
    if USE_DRONE_NN:
        try:
            best_drone_model = tf.keras.models.load_model(drone_model_path)
        except:
            from models import create_drone_model
            best_drone_model = create_drone_model()
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    # make screen fullscreen
    #screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption("Drone Escape")
    clock = pygame.time.Clock()
    dungeon, player, drones, exit_rect, player_start = new_level()
    start_rect = pygame.Rect(player_start[0] - TILE_SIZE // 2, player_start[1] - TILE_SIZE // 2, TILE_SIZE, TILE_SIZE)
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
            player.update_keyboard(dt, dungeon)
        if USE_DRONE_NN:
            batch_update_drones(drones, dt, dungeon, player, best_drone_model)
        else:
            for drone in drones:
                drone.update_random(dt, dungeon, drones)
        if exit_rect.collidepoint(float(player.x), float(player.y)):
            dungeon, player, drones, exit_rect, player_start = new_level()
        for drone in drones:
            if distance((player.x, player.y), (drone.x, drone.y)) < (PLAYER_RADIUS + DRONE_RADIUS):
                player.x, player.y = player.spawn
                player.vx, player.vy = 0, 0
                for drone in drones:
                    drone.x, drone.y = drone.spawn
                    drone.vx, drone.vy = 0, 0
                break
        screen.fill((0, 0, 0))
        for ty in range(MAP_HEIGHT):
            for tx in range(MAP_WIDTH):
                color = COLOR_FLOOR if dungeon[ty, tx] == 1 else COLOR_WALL
                pygame.draw.rect(screen, color, (tx * TILE_SIZE, ty * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        pygame.draw.rect(screen, COLOR_EXIT, exit_rect)
        pygame.draw.rect(screen, COLOR_START, start_rect)
        player.draw(screen)
        for drone in drones:
            drone.draw(screen)
        pygame.display.flip()
    pygame.quit()


def draw_exit_arrow(screen, player_x, player_y, exit_x, exit_y):
    """ Draw an arrow at the edge of the screen pointing toward the exit. """
    import math
    ARROW_SIZE = 15

    # Calculate direction to exit
    dx, dy = exit_x - player_x, exit_y - player_y
    angle = math.atan2(dy, dx)

    # Calculate position at screen edge
    max_x = SCREEN_WIDTH // 2 - ARROW_SIZE
    max_y = SCREEN_HEIGHT // 2 - ARROW_SIZE
    scale_factor = min(max_x / abs(dx) if dx else max_x, max_y / abs(dy) if dy else max_y)

    edge_x = SCREEN_WIDTH // 2 + dx * scale_factor
    edge_y = SCREEN_HEIGHT // 2 + dy * scale_factor

    # Keep the arrow inside the screen
    edge_x = max(ARROW_SIZE, min(SCREEN_WIDTH - ARROW_SIZE, edge_x))
    edge_y = max(ARROW_SIZE, min(SCREEN_HEIGHT - ARROW_SIZE, edge_y))

    # Calculate arrow triangle points
    tip = (edge_x, edge_y)
    left = (edge_x - ARROW_SIZE * math.cos(angle - math.pi / 6),
            edge_y - ARROW_SIZE * math.sin(angle - math.pi / 6))
    right = (edge_x - ARROW_SIZE * math.cos(angle + math.pi / 6),
             edge_y - ARROW_SIZE * math.sin(angle + math.pi / 6))

    pygame.draw.polygon(screen, COLOR_EXIT, [tip, left, right])


def run_manual_mode(USE_PLAYER_NN=True, USE_DRONE_NN=True, path="../models_ga/"):
    if path[-1] != '/':
        path += '/'
    player_model_path = path + "player.keras"
    drone_model_path = path + "drone.keras"
    if USE_PLAYER_NN:
        try:
            best_player_model = tf.keras.models.load_model(player_model_path)
        except:
            from models import create_player_model
            best_player_model = create_player_model()
    if USE_DRONE_NN:
        try:
            best_drone_model = tf.keras.models.load_model(drone_model_path)
        except:
            from models import create_drone_model
            best_drone_model = create_drone_model()

    pygame.init()
    infoObject = pygame.display.Info()
    screen_w, screen_h = infoObject.current_w, infoObject.current_h
    print (screen_w, screen_h)
    screen = pygame.display.set_mode((screen_w, screen_h))
    #screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Drone Escape")
    clock = pygame.time.Clock()

    dungeon, player, drones, exit_rect, player_start = new_level()

    # Camera Offsets
    camera_x, camera_y = player.x - SCREEN_WIDTH // 2, player.y - SCREEN_HEIGHT // 2

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # Convert to seconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        goal = (exit_rect.x + TILE_SIZE / 2, exit_rect.y + TILE_SIZE / 2)

        if USE_PLAYER_NN:
            player.update_nn(dt, dungeon, drones, goal, best_player_model)
        else:
            player.update_keyboard(dt, dungeon)

        if USE_DRONE_NN:
            batch_update_drones(drones, dt, dungeon, player, best_drone_model)
        else:
            for drone in drones:
                drone.update_random(dt, dungeon, drones)

        # Adjust Camera Offset to keep player centered, but fix to the screen edges if player moves to the edge
        # Adjust Camera Offset to keep player centered
        camera_x = player.x - SCREEN_WIDTH // 2
        camera_y = player.y - SCREEN_HEIGHT // 2

        # Clamp camera position so it doesn’t move beyond the dungeon bounds
        max_camera_x = MAP_WIDTH * TILE_SIZE - screen_w // 2
        max_camera_y = MAP_HEIGHT * TILE_SIZE - screen_h // 2
        if camera_x < 0:
            camera_x = 0
        elif camera_x > max_camera_x:
            camera_x = max_camera_x
        if camera_y < 0:
            camera_y = 0
        elif camera_y > max_camera_y:
            camera_y = max_camera_y

        camera_x = max(0, min(camera_x, max_camera_x))
        camera_y = max(0, min(camera_y, max_camera_y))

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
        for ty in range(MAP_HEIGHT):
            for tx in range(MAP_WIDTH):
                color = COLOR_FLOOR if dungeon[ty, tx] == 1 else COLOR_WALL
                tile_x = tx * TILE_SIZE - camera_x
                tile_y = ty * TILE_SIZE - camera_y
                if -1 <= tile_x < SCREEN_WIDTH + 1 and -1 <= tile_y < SCREEN_HEIGHT + 1:
                    pygame.draw.rect(screen, color, (tile_x, tile_y, TILE_SIZE, TILE_SIZE))

        if 0 <= exit_rect.x - camera_x < SCREEN_WIDTH and 0 <= exit_rect.y - camera_y < SCREEN_HEIGHT:
            pygame.draw.rect(screen, COLOR_EXIT,
                             (exit_rect.x - camera_x, exit_rect.y - camera_y, TILE_SIZE, TILE_SIZE))
        else:
            draw_exit_arrow(screen, player.x, player.y, exit_rect.x + TILE_SIZE / 2, exit_rect.y + TILE_SIZE / 2)

        # Draw Player (always at the center of screen)
        if 0 <= player.x - camera_x < SCREEN_WIDTH and 0 <= player.y - camera_y < SCREEN_HEIGHT:
            pygame.draw.circle(screen, COLOR_PLAYER,
                               (int(player.x - camera_x), int(player.y - camera_y)), PLAYER_RADIUS)

        # Draw Drones
        for drone in drones:
            drone_x = int(drone.x - camera_x)
            drone_y = int(drone.y - camera_y)
            if 0 <= drone_x < SCREEN_WIDTH and 0 <= drone_y < SCREEN_HEIGHT:
                pygame.draw.circle(screen, COLOR_DRONE, (drone_x, drone_y), DRONE_RADIUS)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    run_manual_mode(False, False)
