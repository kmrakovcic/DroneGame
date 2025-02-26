import os
import random
import pygame
import numpy as np
from config import TILE_SIZE, MAP_WIDTH, MAP_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT, DRONE_NUMBER
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
    dungeon = [[0 for _ in range(MAP_WIDTH+2)] for _ in range(MAP_HEIGHT+2)]
    rooms = []
    from config import ROOM_MIN_SIZE, ROOM_MAX_SIZE, MAX_ROOMS  # import here if needed
    for _ in range(MAX_ROOMS):
        w = random.randint(ROOM_MIN_SIZE, ROOM_MAX_SIZE)
        h = random.randint(ROOM_MIN_SIZE, ROOM_MAX_SIZE)
        x = random.randint(1, MAP_WIDTH - w - 2)
        y = random.randint(1, MAP_HEIGHT - h - 2)
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
        start_tile = np.array([MAP_WIDTH//2, MAP_HEIGHT//2])
        start_angle = random.choice(["top-left", "top-right", "bottom-left", "bottom-right"])
        if start_angle == "top-left":
            # find the room that is most to the top left corner
            start_room = min(rooms, key=lambda r: distance([0, 0], [r.center[0], r.center[1]]))
            # start from the wall closest to the edge of the screen
            if start_room.x1 < start_room.y1:
                start_tile = np.array([start_room.x1-1, start_room.center[1]])
            else:
                start_tile = np.array([start_room.center[0], start_room.y1-1])
        elif start_angle == "top-right":
            # find the room that is most to the top right corner
            start_room = min(rooms, key=lambda r: distance([MAP_WIDTH, 0], [r.center[0], r.center[1]]))
            # start from the wall closest to the edge of the screen
            if MAP_WIDTH - start_room.x2-1 < start_room.y1:
                start_tile = np.array([start_room.x2, start_room.center[1]])
            else:
                start_tile = np.array([start_room.center[0], start_room.y1-1])
        elif start_angle == "bottom-left":
            # find the room that is most to the bottom left corner
            start_room = min(rooms, key=lambda r: distance([0, MAP_HEIGHT], [r.center[0], r.center[1]]))
            # start from the wall closest to the edge of the screen
            if start_room.x1 < MAP_HEIGHT - start_room.y2 - 1:
                start_tile = np.array([start_room.x1-1, start_room.center[1]])
            else:
                start_tile = np.array([start_room.center[0], start_room.y2])
        elif start_angle == "bottom-right":
            # find the room that is most to the bottom right corner
            start_room = min(rooms, key=lambda r: distance([MAP_WIDTH, MAP_HEIGHT], [r.center[0], r.center[1]]))
            # start from the wall closest to the edge of the screen
            if MAP_WIDTH - start_room.x2 - 1 < MAP_HEIGHT - start_room.y2 - 1:
                start_tile = np.array([start_room.x2, start_room.center[1]])
            else:
                start_tile = np.array([start_room.center[0], start_room.y2])
        player_start = (start_tile[0] * TILE_SIZE + TILE_SIZE / 2, start_tile[1] * TILE_SIZE + TILE_SIZE / 2)
        player = Player(*player_start)
        if len(rooms) > 1:
            rooms_distance = [distance(start_room.center, r.center) for r in rooms if r != start_room]
            exit_room = [r for r in rooms if r != start_room][np.argsort(rooms_distance)[-1]]
        else:
            exit_room = start_room

        if exit_room.x1 < MAP_WIDTH - exit_room.x2:
            exit_tile = np.array([exit_room.x1 - 1, exit_room.center[1]])
            if dungeon [exit_tile[1]][exit_tile[0]] == 0:
                exit_tile = np.array([exit_room.x2, exit_room.center[1]])
        else:
            exit_tile = np.array([exit_room.x2, exit_room.center[1]])
        dungeon[exit_tile[1]][exit_tile[0]] = 1
        dungeon[start_tile[1]][start_tile[0]] = 1
        dungeon_np = np.array(dungeon, dtype=np.int32)
        exit_rect = pygame.Rect(exit_tile[0] * TILE_SIZE, exit_tile[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    else:
        player_start = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
        player = Player(*player_start)
        exit_rect = pygame.Rect(SCREEN_WIDTH - TILE_SIZE, SCREEN_HEIGHT - TILE_SIZE, TILE_SIZE, TILE_SIZE)
        start_room = None

    drones = []
    num_drones = DRONE_NUMBER
    for dn in range(num_drones):
        while True:
            tx = random.randint(0, MAP_WIDTH - 1)
            ty = random.randint(0, MAP_HEIGHT - 1)
            if dungeon_np[ty, tx] == 1:
                if start_room is not None:
                    if start_room.x1 <= tx < start_room.x2 and start_room.y1 <= ty < start_room.y2:
                        continue
                tile_center = (tx * TILE_SIZE + TILE_SIZE / 2, ty * TILE_SIZE + TILE_SIZE / 2)
                if distance(tile_center, player_start) > TILE_SIZE and not exit_rect.collidepoint(tile_center):
                    if not any(distance(tile_center, (d.x, d.y)) < TILE_SIZE for d in drones):
                        d = Drone(tile_center[0], tile_center[1])
                        d.spawn = tile_center
                        drones.append(d)
                        break
    player.spawn = player_start
    return dungeon_np, player, drones, exit_rect, player_start

if __name__ == "__main__":
    os.makedirs("../levels/", exist_ok=True)
    for i in range(10):
        dungeon_np, player, drones,exit_rect, player_start = new_level()
        dungeon_expanded = np.kron(np.array(dungeon_np, dtype=np.int32), np.ones((TILE_SIZE, TILE_SIZE), dtype=np.int32))
        # save dugeon to csv
        np.savetxt('../levels/dungeon'+str(i+1)+'.csv', dungeon_expanded, delimiter=',', fmt='%d')