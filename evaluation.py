import os

import torch
from torch import nn

from model import get_res_pre_trained


def load_model(args):
    model = get_res_pre_trained(model_name=args.MODEL)
    model.load_state_dict(torch.load(args.MODEL_PATH))
    model = model.to(args.device)
    return model


def save_model(args, model):
    torch.save(model.state_dict(), args.MODEL_PATH)

    # 나중에 사용해 봅시다.
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': loss,
    #     'loss_val_avg':loss_val_avg
    #     }, check_path)


if __name__ == "__main__":
    pass
