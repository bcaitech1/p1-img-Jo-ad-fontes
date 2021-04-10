import os
import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description="Mask!!!")
    parser.add_argument("--EPOCHS", default=10, type=int)
    parser.add_argument("--BATCH_SIZE", default=64, type=int)
    parser.add_argument("--LEARNING_RATE", default=0.001, type=float)
    parser.add_argument("--MODEL", default="resnet50", type=str)
    parser.add_argument("--TRAIN_KEY", default="classifier", type=str)
    parser.add_argument("--MODEL_PATH", default="/opt/ml/code/checkpoint/", type=str)

    # cross_entropy, focal, label_smoothing, f1
    parser.add_argument("--LOSS", default="cross_entropy", type=str)

    args = parser.parse_args()

    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.MODEL_PATH, exist_ok=True)

    MODEL_PATH = os.path.join(args.MODEL_PATH, args.MODEL)
    args.MODEL_PATH = f"{MODEL_PATH}_{args.TRAIN_KEY}.pth"

    return args


if __name__ == "__main__":
    pass
