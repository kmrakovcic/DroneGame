# main.py
import argparse
from game import run_manual_mode
import training_cma
import training_ga

def main():
    parser = argparse.ArgumentParser(description="Run game or training modes")
    parser.add_argument('--mode', type=str, choices=['manual', 'train_cma', 'train_ga'], default='manual')
    args = parser.parse_args()

    if args.mode == 'manual':
        run_manual_mode(USE_PLAYER_NN=False, USE_DRONE_NN=True, path="../models_ga/")
    elif args.mode == 'train_cma':
        training_cma.run_training("../models_cma/", use_parallel_evaluation=True)
    elif args.mode == 'train_ga':
        training_ga.run_training("../models_ga/", use_parallel_evaluation=True)

if __name__ == "__main__":
    main()
