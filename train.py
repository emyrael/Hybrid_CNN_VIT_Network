from tqdm import tqdm
import sys, os
import torch
import json
import argparse
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models.model import VitConvNet
from utils import get_augmentation

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


def main(args):

    # create the folder for logging files
    path_to_log = args.log_path
    os.makedirs(path_to_log)
    txt_log = os.path.join(path_to_log, "log.txt")
    text_log = open(txt_log, "w")
    img_log = os.path.join(path_to_log, "training_curves.png")
    
    # cuda or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # models definitions

    with open(args.cfg_path, "r") as f:
        model_cfg = json.load(f)

    model = VitConvNet(model_cfg["backbone_cfg"], model_cfg["vit_cfg"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.to(device)

    # dataloaders

    transformer = get_augmentation(do_hflip=False, do_vflip=False)

    train_loader = DataLoader(
        ImageFolder(args.train_data, transform=transformer),
        batch_size=args.bs, shuffle=True
    )

    test_loader = DataLoader(
        ImageFolder(args.test_data, transform=transformer),
        batch_size=args.bs, shuffle=True
    )

    # training process

    train_loss_list = []
    test_loss_list = []

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        train_pred = []
        train_label = []

        for data, label in tqdm(train_loader):
            data = data.float()
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_pred.append(output.cpu().detach().numpy())
            train_label.append(nn.functional.one_hot(label).cpu().detach().numpy())

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)


        test_pred = []
        test_label = []

        with torch.no_grad():
            epoch_test_accuracy = 0
            epoch_test_loss = 0
            for data, label in test_loader:
                data = data.float()
                data = data.to(device)
                label = label.to(device)

                test_output = model(data)
                test_loss = criterion(test_output, label)

                test_pred.append(test_output.cpu().detach().numpy())
                test_label.append(nn.functional.one_hot(label).cpu().detach().numpy())

                epoch_test_loss += test_loss / len(test_loader)
        
        # Compute all metrics etc.

        train_pred, test_pred = np.concatenate(train_pred, axis = 0), np.concatenate(test_pred, axis = 0)
        train_label, test_label = np.concatenate(train_label, axis = 0), np.concatenate(test_label, axis = 1\0)
        
        epoch_accuracy = accuracy_score(np.argmax(train_label, axis = 1), np.argmax(train_pred, axis = 1))
        epoch_test_accuracy = accuracy_score(np.argmax(test_label, axis = 1), np.argmax(test_pred, axis = 1))

        epoch_auc = roc_auc_score(train_label, train_pred)
        epoch_val_auc = roc_auc_score(test_label, test_pred)
        
        epoch_msg = f"""
        -------------------------|Epoch : {epoch + 1}|----------------------------------\n
        loss : {epoch_loss:.4f} | acc: {epoch_accuracy:.4f} | roc_auc: {epoch_auc:.4f} \n
        val_loss : {epoch_test_loss:.4f} | val_acc: {epoch_test_accuracy:.4f} | val_roc_auc: {epoch_val_auc:.4f} \n"""

        if args.print_confusion_matrix:
            epoch_msg += f"""
            confusion_matrix:\n {confusion_matrix(np.argmax(train_label, axis = 1), np.argmax(train_pred, axis = 1))}\n
            val_confusion_matrix:\n {confusion_matrix(np.argmax(test_label, axis = 1), np.argmax(test_pred, axis = 1))}\n
            """
        print(epoch_msg)

        text_log.write(epoch_msg)
        model.train()

        train_loss_list.append(epoch_loss)
        test_loss_list.append(epoch_test_loss)

        fig, ax = plt.figure(figsize = (15, 15))
        ax.plot(list(range(len(train_loss_list))), train_loss_list, label = "train loss")
        ax.plot(list(range(len(test_loss_list))), test_loss_list, label = "test loss")
        
        fig.savefig(img_log)
        plt.close(fig)




    torch.save(model, os.path.join(args.msp, args.dsname))


if __name__ is "__main__":
    desc = "Vision Transformer"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training')

    parser.add_argument('--bs', type=int, default=64, help='batch size')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for Adam optimizer')

    parser.add_argument('--ps', type=int, default=16, help='Path_size')

    parser.add_argument('--cfg_path', type=str, default="cfg.json", help='path to config')

    parser.add_argument('--train_data', type=str, help='path_to_train_data')

    parser.add_argument('--test_data', type=str, help='path_to_test_data')

    parser.add_argument('--dsname', type=str, help='dataset name')

    parser.add_argument('--msp', type=str, help='path_to_save_model')

    parser.add_argument('--log_path', type=str, help="path to the folder with logging files")

    parser.add_argument('--print_confusion_matrix', type=bool, default=False, help="Do you need confusion matrix in output? It takes much place")

    args = parser.parse_args()

    main(args)
