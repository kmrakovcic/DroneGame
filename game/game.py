import pygame
import numpy as np
import time
import random
import math
from config import MAP_HEIGHT, MAP_WIDTH, TILE_SIZE, COLOR_FLOOR, COLOR_WALL, COLOR_EXIT, PLAYER_RADIUS, DRONE_RADIUS
from level import new_level
from entities.player import Player
from entities.drone import batch_update_drones, Drone
from utils import distance
from simulation import simulate_game

def run_manual_mode(USE_PLAYER_NN=True, USE_DRONE_NN=True, path="../models_ga/"):
    from config import SCREEN_WIDTH, SCREEN_HEIGHT, TILE_SIZE
    from entities.player import Player
    from entities.drone import Drone, batch_update_drones
    import tensorflow as tf

    if path[-1]!='/':
        path+='/'
    player_model_path = path+"player.keras"
    drone_model_path = path+"drone.keras"
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
    pygame.display.set_caption("Drone Escape")
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

if __name__ == "__main__":
    run_manual_mode(False, False)
