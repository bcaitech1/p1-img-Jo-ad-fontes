import os
import random
import time

import wandb
import torch
import numpy as np

from tqdm import tqdm
from torch import nn, optim


from config import get_args
from dataset import get_loader
from evaluation import save_model, load_model
from loss import create_criterion
from model import get_res_pre_trained
from metrics import (
    get_scores,
    get_confusion_matrix,
    tensor_to_numpy,
    tensor_2d_to_1d,
    get_all_datas,
)
from gpu import gpu


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


seed_everything(42)


def train(args, model, criterion, optimizer, dataloader):
    print("train")
    model.train()
    epoch_loss = 0
    labels = torch.tensor([]).to(args.device)
    preds = torch.tensor([]).to(args.device)

    for batch_img, batch_lab in dataloader:
        optimizer.zero_grad()

        inputs = batch_img.to(args.device)
        targets = batch_lab.to(args.device)

        outputs = model.forward(inputs)
        loss = criterion(outputs, targets)

        # calculate score
        labels = torch.cat((labels, targets))
        preds = torch.cat((preds, outputs))

        loss.backward()

        optimizer.step()

        epoch_loss += loss

    labels = labels.reshape(-1).detach().cpu().numpy()
    preds = torch.argmax(preds, dim=1).detach().cpu().numpy()

    f1_sco, pr_sco, re_sco, acc = get_scores(labels, preds)

    return (epoch_loss / len(dataloader)), [f1_sco, pr_sco, re_sco, acc]


def evaluate(args, model, criterion, dataloader):
    print("eval")
    model.eval()
    epoch_loss = 0

    labels = torch.tensor([]).to(args.device)
    preds = torch.tensor([]).to(args.device)

    with torch.no_grad():
        for batch_img, batch_lab in dataloader:

            inputs = batch_img.to(args.device)
            targets = batch_lab.to(args.device)

            outputs = model.forward(inputs)

            # calculate score
            labels = torch.cat((labels, targets))
            preds = torch.cat((preds, outputs))

            loss = criterion(outputs, targets)
            epoch_loss += loss

    labels = labels.reshape(-1).detach().cpu().numpy()
    preds = torch.argmax(preds, dim=1).detach().cpu().numpy()

    f1_sco, pr_sco, re_sco, acc = get_scores(labels, preds)

    return (epoch_loss / len(dataloader)), [f1_sco, pr_sco, re_sco, acc]


def run(args, model, criterion, optimizer, train_loader, val_loader):
    best_valid_loss = float("inf")
    best_f1_score = 0

    train_scores = []
    val_scores = []

    for epoch in range(args.EPOCHS):

        train_loss, train_scores = train(
            args, model, criterion, optimizer, train_loader
        )
        valid_loss, val_scores = evaluate(args, model, criterion, val_loader)

        if val_scores[0] > best_f1_score:
            best_f1_score = val_scores[0]
            save_model(args, model)

        wandb.log({"train_loss": train_loss, "valid_loss": valid_loss})
        wandb.log(
            {
                "f1 score": val_scores[0],
                "precision": val_scores[1],
                "recall": val_scores[2],
                "acc": val_scores[3],
            }
        )

        print(f"epoch:{epoch+1}/{args.EPOCHS}")

        print(f"train_loss: {train_loss:.4f} valid_loss: {valid_loss:.4f}")

        scores = ["F1 score", "Precision score", "Recall score", "Acc score"]
        for i, (t_s, v_s) in enumerate(zip(train_scores, val_scores)):
            print(f"TRAIN {scores[i]}: {t_s:.4f} VAL {scores[i]}: {v_s:.4f}")


def main(args):
    wandb.init(project="stage-1", reinit=True)
    wandb.run.name = args.MODEL
    wandb.config.update(args)

    args = wandb.config

    train_loader, val_loader = get_loader(args.BATCH_SIZE)
    print("Get loader")
    model = get_res_pre_trained(args.MODEL).to(args.device)
    print("Load model")

    wandb.watch(model)

    criterion = create_criterion(args.LOSS)
    optimizer = optim.Adam(model.parameters(), lr=args.LEARNING_RATE)
    print("Run")
    run(args, model, criterion, optimizer, train_loader, val_loader)


if __name__ == "__main__":
    args = get_args()
    torch.cuda.empty_cache()
    print("PyTorch version:[%s]." % (torch.__version__))
    print("This code use [%s]." % (args.device))

    main(args)
