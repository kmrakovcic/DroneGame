# config.py
import os

# Disable GPU usage (or configure as needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Screen and tile configuration
TILE_SIZE = 16
MAP_WIDTH = 60    # in tiles
MAP_HEIGHT = 40   # in tiles
SCREEN_WIDTH = MAP_WIDTH * TILE_SIZE
SCREEN_HEIGHT = MAP_HEIGHT * TILE_SIZE

DRONE_NUMBER = 6

# Movement speeds
PLAYER_SPEED = 50.0
DRONE_SPEED = 50.0

# Radii for collision detection
PLAYER_RADIUS = 6
DRONE_RADIUS = 6

# Colors (RGB)
COLOR_WALL = (100, 100, 100)
COLOR_FLOOR = (200, 200, 200)
COLOR_PLAYER = (50, 200, 50)
COLOR_DRONE = (200, 50, 50)
COLOR_EXIT = (50, 50, 200)
COLOR_START = (50, 50, 50)

# Dungeon generation parameters
ROOM_MAX_SIZE = 8
ROOM_MIN_SIZE = 4
MAX_ROOMS = 15
