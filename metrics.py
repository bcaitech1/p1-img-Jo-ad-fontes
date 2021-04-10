import torch
import pandas as pd

from evaluation import load_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# 구건모님의 도움


def tensor_2d_to_1d(tensor):
    """ 2d, 1d """
    if tensor.ndim == 2:
        return tensor.reshape(-1)
    return tensor


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def get_all_datas(args, model, dataloader):
    """ dataloader: validation """

    all_images = torch.tensor([]).to(args.device)
    all_labels = torch.tensor([]).to(args.device)
    all_preds = torch.tensor([]).to(args.device)

    for images, labels in dataloader:
        images, labels = images.to(args.device), labels.to(args.device)

        preds = model(images)

        all_images = torch.cat((all_images, images))
        all_labels = torch.cat((all_labels, labels))
        all_preds = torch.cat((all_preds, preds))

    all_labels = tensor_2d_to_1d(all_labels)
    return all_images, all_labels, all_preds


def get_scores(labels, preds):
    """
    labels: numpy
    preds: numpy, after argmax
    """
    f1_sco = f1_score(labels, preds, average="macro")
    pr_sco = precision_score(labels, preds, average="macro")
    re_sco = recall_score(labels, preds, average="macro")
    accuracy = accuracy_score(labels, preds)

    return f1_sco, pr_sco, re_sco, accuracy


def get_confusion_matrix(labels, preds):
    """
    labels: numpy
    preds: numpy, after argmax
    """
    cm = confusion_matrix(labels, preds)
    return cm


from dataset import get_loader
from config import get_args
from evaluation import load_model


if __name__ == "__main__":
    pass
