# main.py
import argparse
from game import run_manual_mode

def main():
    parser = argparse.ArgumentParser(description="Run game or training modes")
    parser.add_argument('--mode', type=str,
                        choices=['manual', 'train_cma', 'train_ga', 'train_nes'],
                        default='manual')
    parser.add_argument('--epochs', type=int, default=1000)
    args = parser.parse_args()

    if args.mode == 'manual':
        run_manual_mode(USE_PLAYER_NN=True, USE_DRONE_NN=True, path="../models_nes/")
    elif args.mode == 'train_cma':
        import training_cma
        training_cma.run_training("../models_cma/", epochs=args.epochs, use_parallel_evaluation=True)
    elif args.mode == 'train_ga':
        import training_ga
        training_ga.run_training("../models_ga/", epochs=args.epochs, use_parallel_evaluation=True)
    elif args.mode == 'train_nes':
        import training_nes
        training_nes.run_training("../models_nes/", epochs=args.epochs, use_parallel_evaluation=True)

if __name__ == "__main__":
    main()
