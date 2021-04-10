from dataclasses import dataclass

import argparse
import torch


@dataclass
class Config:
    device: str
    ROOT: str
    EPOCHS: int
    BATCH_SIZE: int
    LEARNING_RATE: float


def get_config():
    parser = argparse.ArgumentParser(description="Mask!!!")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=0.001, type=float)

    args = parser.parse_args()

    config = Config(
        EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        LEARNING_RATE=args.lr,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        ROOT="./input/data/train/images",
    )

    return config
