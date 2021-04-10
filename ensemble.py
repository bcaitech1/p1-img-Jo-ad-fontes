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
    save_path = f"/opt/ml/code/checkpoint/vote_list/{model_name}"

    model.load_state_dict(torch.load(save_path))
    print(f"Loaded model:{model_name}")

    return model


def get_eval_df(output_np):
    eval_loader = get_eval_loader()

    img_names = np.array([])
    img_classes = output_np
    df = pd.DataFrame()

    for batch_img_name, batch_img in tqdm(eval_loader):
        inputs = batch_img

        for img_name in batch_img_name:
            img_name = img_name.split("/")[-1]
            img_names = np.append(img_names, img_name)

    df["ImageID"] = img_names
    df["ans"] = img_classes

    return df


def get_output_tensor(model: nn.Module, device: str):
    model.to(device)
    model.eval()

    eval_loader = get_eval_loader()

    with torch.no_grad():
        output_tensor = torch.tensor([]).to(device)
        for batch_img_name, batch_img in tqdm(eval_loader):
            inputs = batch_img.to(device)
            outputs = model(inputs)
            output_tensor = torch.cat((output_tensor, outputs), dim=0)

    return output_tensor


def to_csv(df: pd.DataFrame, model_name: str):
    save_folder = f"/opt/ml/code/result/"
    os.makedirs(save_folder, exist_ok=True)

    _submission = pd.read_csv("/opt/ml/input/data/eval/info.csv")
    _submission.drop("ans", axis=1, inplace=True)

    df = pd.merge(_submission, df, left_on="ImageID", right_on="ImageID", how="inner")
    file_path = os.path.join(save_folder, f"final_submission.csv")

    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    print("Start")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_paths = os.listdir("/opt/ml/code/checkpoint/vote_list")

    output_tensor_add = torch.zeros(12600, 18).to(device)

    for mp in model_paths:
        if mp.endswith(".pth"):
            model = load_model(mp)
            output_tensor = get_output_tensor(model, device)
            output_tensor_add = torch.add(output_tensor_add, output_tensor)

    output_np = torch.argmax(output_tensor_add, dim=1).detach().cpu().numpy()
    df = get_eval_df(output_np)
    to_csv(df, mp[:-4])

    print("Finish")
