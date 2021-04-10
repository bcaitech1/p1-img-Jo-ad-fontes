from dataset import get_loader
from config import get_config
from evaluation import func_eval
from evaluation import save_model

from model import get_res_pre_trained
from tqdm import tqdm

from torch import nn, optim

import torch
import os

cfg = get_config()


def train_model(model: nn.Module, train_loader, eval_loader, EPOCHS: int, device: str):
    # Training Phase
    print_every = 1
    print("Start training !")
    model.train()

    # Training loop
    loss_val_sum = 1
    for epoch in range(EPOCHS):
        pre_loss = loss_val_sum
        loss_val_sum = 0
        for batch_img, batch_lab in tqdm(train_loader):
            inputs = batch_img.to(device)
            targets = batch_lab.to(device)

            # Inference & Calculate loss
            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val_sum += loss
        if ((epoch % print_every) == 0) or (epoch == (EPOCHS - 1)):
            loss_val_avg = loss_val_sum / len(train_loader)

            train_accr = func_eval(model, train_loader, device)
            test_accr = func_eval(model, eval_loader, device)
            print(
                f"epoch:[{epoch+1}/{EPOCHS}] loss:[{loss_val_avg:.3f}] train_accr:[{train_accr:.3f}] test_accuracy:[{test_accr:.3f}]"
            )
            # print(f'epoch:[{epoch+1}/{EPOCHS}] loss:[{loss_val_avg:.3f}]')
            # print(loss_val_sum)
        if (loss_val_sum < pre_loss) or epoch == (EPOCHS - 1):
            save_model(model, cfg.MODEL, epoch, loss_val_avg)

    print("Training Done !")


if __name__ == "__main__":
    print("PyTorch version:[%s]." % (torch.__version__))
    print("This code use [%s]." % (cfg.device))

    train_loader, eval_loader = get_loader(cfg.BATCH_SIZE)

    # print(cfg.EPOCHS)
    # error : num_samples should be a positive integer value, but got num_samples=0

    model = get_res_pre_trained(cfg.MODEL).to(cfg.device)
    # print(model)
    # images, labels = next(iter(train_loader))
    # print(f'images shape: {images.shape}')
    # print(f'labels shape: {labels.shape}')
    # print(images)
    # print(labels)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    train_model(model, train_loader, eval_loader, cfg.EPOCHS, cfg.device)
