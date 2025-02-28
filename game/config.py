# config.py
import os

# Disable GPU usage (or configure as needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ZOOM_IN_LVL = 1
# Screen and tile configuration
TILE_SIZE = 16 * ZOOM_IN_LVL
MAP_WIDTH = 30   # in tiles
MAP_HEIGHT = 20  # in tiles
SCREEN_WIDTH = MAP_WIDTH * TILE_SIZE
SCREEN_HEIGHT = MAP_HEIGHT * TILE_SIZE

DRONE_NUMBER = 3

# Movement speeds
PLAYER_SPEED = 50.0 * ZOOM_IN_LVL
DRONE_SPEED = 50.0 * ZOOM_IN_LVL

# Radii for collision detection
PLAYER_RADIUS = 5 * ZOOM_IN_LVL
DRONE_RADIUS = 5 * ZOOM_IN_LVL

# Colors (RGB)
COLOR_WALL = (100, 100, 100)
COLOR_FLOOR = (200, 200, 200)
COLOR_PLAYER = (50, 200, 50)
COLOR_DRONE = (50, 50, 50)
COLOR_EXIT = (50, 50, 200)
COLOR_START = (50, 100, 100)

# Dungeon generation parameters
ROOM_MAX_SIZE = 8
ROOM_MIN_SIZE = 4
MAX_ROOMS = 15
