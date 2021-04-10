import os

import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from torch import nn
from model import get_res_pre_trained
from dataset import get_eval_loader
from dataset import eval_data
from config import get_args

args = get_args()


def load_model(model_name: str):
    base = model_name.split("_")[0]
    model = get_res_pre_trained(base)
    save_path = f"/opt/ml/code/checkpoint/last/{model_name}"

    model.load_state_dict(torch.load(save_path))
    print(f"Loaded model:{model_name}")

    return model


def get_eval_df(model: nn.Module, device: str):
    model.to(device)
    model.eval()
    eval_loader = get_eval_loader()

    with torch.no_grad():
        img_names = np.array([])
        img_classes = np.array([])
        df = pd.DataFrame()

        for batch_img_name, batch_img in tqdm(eval_loader):
            inputs = batch_img.to(device)
            outputs = model(inputs)
            _, predict_targets = torch.max(outputs.data, 1)

            for img_name in batch_img_name:
                img_name = img_name.split("/")[-1]
                img_names = np.append(img_names, img_name)

            img_classes = np.append(img_classes, predict_targets.to("cpu").numpy())

        df["ImageID"] = img_names
        df["ans"] = img_classes

    return df


def to_csv(df: pd.DataFrame, model_name: str):
    save_folder = f"/opt/ml/code/result/"
    os.makedirs(save_folder, exist_ok=True)

    _submission = pd.read_csv("/opt/ml/input/data/eval/info.csv")
    _submission.drop("ans", axis=1, inplace=True)

    df = pd.merge(_submission, df, left_on="ImageID", right_on="ImageID", how="inner")

    file_path = os.path.join(save_folder, f"{model_name}_submission.csv")

    df.to_csv(file_path, index=False)


def get_model_path():
    pass


if __name__ == "__main__":
    print("Start")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_paths = os.listdir("/opt/ml/code/checkpoint/last")
    for mp in model_paths:
        if mp.endswith(".pth"):
            model = load_model(mp)
            df = get_eval_df(model, device)
            to_csv(df, mp[:-4])

    print("Finish")
