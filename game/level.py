import os
import random
import pygame
import numpy as np
from config import TILE_SIZE, MAP_WIDTH, MAP_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT
from entities.player import Player
from entities.drone import Drone
from utils import distance

class Room:
    def __init__(self, x, y, w, h):
        self.x1 = x
        self.y1 = y
        self.x2 = x + w
        self.y2 = y + h
        self.center = ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def intersects(self, other):
        return (self.x1 <= other.x2 and self.x2 >= other.x1 and
                self.y1 <= other.y2 and self.y2 >= other.y1)

def generate_dungeon():
    dungeon = [[0 for _ in range(MAP_WIDTH)] for _ in range(MAP_HEIGHT)]
    rooms = []
    from config import ROOM_MIN_SIZE, ROOM_MAX_SIZE, MAX_ROOMS  # import here if needed
    for _ in range(MAX_ROOMS):
        w = random.randint(ROOM_MIN_SIZE, ROOM_MAX_SIZE)
        h = random.randint(ROOM_MIN_SIZE, ROOM_MAX_SIZE)
        x = random.randint(0, MAP_WIDTH - w - 1)
        y = random.randint(0, MAP_HEIGHT - h - 1)
        new_room = Room(x, y, w, h)
        if any(new_room.intersects(r) for r in rooms):
            continue
        for i in range(new_room.x1, new_room.x2):
            for j in range(new_room.y1, new_room.y2):
                dungeon[j][i] = 1
        if rooms:
            prev_center = rooms[-1].center
            new_center = new_room.center
            if random.randint(0, 1):
                for x_corr in range(min(prev_center[0], new_center[0]), max(prev_center[0], new_center[0]) + 1):
                    dungeon[prev_center[1]][x_corr] = 1
                for y_corr in range(min(prev_center[1], new_center[1]), max(prev_center[1], new_center[1]) + 1):
                    dungeon[y_corr][new_center[0]] = 1
            else:
                for y_corr in range(min(prev_center[1], new_center[1]), max(prev_center[1], new_center[1]) + 1):
                    dungeon[y_corr][prev_center[0]] = 1
                for x_corr in range(min(prev_center[0], new_center[0]), max(prev_center[0], new_center[0]) + 1):
                    dungeon[new_room.center[1]][x_corr] = 1
        rooms.append(new_room)
    return dungeon, rooms

def new_level():
    dungeon, rooms = generate_dungeon()
    dungeon_np = np.array(dungeon, dtype=np.int32)
    if rooms:
        player_room = random.choice(rooms)
        start_tile = player_room.center
        player_start = (start_tile[0] * TILE_SIZE + TILE_SIZE / 2, start_tile[1] * TILE_SIZE + TILE_SIZE / 2)
        player = Player(*player_start)
        if len(rooms) > 1:
            possible_exits = [r for r in rooms if r != player_room]
            exit_room = random.choice(possible_exits)
        else:
            exit_room = player_room
        exit_tile = exit_room.center
        exit_rect = pygame.Rect(exit_tile[0] * TILE_SIZE, exit_tile[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    else:
        player_start = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
        player = Player(*player_start)
        exit_rect = pygame.Rect(SCREEN_WIDTH - TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE, TILE_SIZE, TILE_SIZE)
        player_room = None

    drones = []
    num_drones = 3
    for dn in range(num_drones):
        while True:
            tx = random.randint(0, MAP_WIDTH - 1)
            ty = random.randint(0, MAP_HEIGHT - 1)
            if dungeon_np[ty, tx] == 1:
                if player_room is not None:
                    if player_room.x1 <= tx < player_room.x2 and player_room.y1 <= ty < player_room.y2:
                        continue
                tile_center = (tx * TILE_SIZE + TILE_SIZE / 2, ty * TILE_SIZE + TILE_SIZE / 2)
                if distance(tile_center, player_start) > TILE_SIZE and not exit_rect.collidepoint(tile_center):
                    if not any(distance(tile_center, (d.x, d.y)) < TILE_SIZE for d in drones):
                        drones.append(Drone(tile_center[0], tile_center[1]))
                        break
    return dungeon_np, player, drones, exit_rect, player_start

if __name__ == "__main__":
    os.makedirs("../levels/", exist_ok=True)
    for i in range(10):
        dungeon_np, player, drones,exit_rect, player_start = new_level()
        dungeon_expanded = np.kron(np.array(dungeon_np, dtype=np.int32), np.ones((TILE_SIZE, TILE_SIZE), dtype=np.int32))
        # save dugeon to csv
        np.savetxt('../levels/dungeon'+str(i+1)+'.csv', dungeon_expanded, delimiter=',', fmt='%d')