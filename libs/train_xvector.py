#!/usr/bin/env python3
import argparse
import logging
import math
import random
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.Xvector import Xvector, Xvector_AttnPooling
from kaldi_data.data_loader import EgsDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(filename)s] %(message)s")
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxceleb-egs", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--feat-dim", type=int, required=True)
    parser.add_argument("--num-spks", type=int, required=True)

    parser.add_argument("--attention", type=bool, default=False)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dropout-p", type=float, default=0.)
    parser.add_argument("--initial-lr", type=float, default=1e-3)
    parser.add_argument("--final-lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    return args


def nnet3_egs_info(egs_dir):
    train_list = glob(f'{egs_dir}/egs.*.ark')
    train_list.sort()
    valid_list = glob(f'{egs_dir}/valid_egs.*.ark')
    valid_list.sort()

    return train_list, valid_list


def train(model, dataloader, criterion, optimizer, iteration):
    model.train()

    total_loss, total_correct = 0., 0
    desc = f'Iter {iteration}'
    for feats, labels in tqdm(dataloader, desc=desc, leave=False, ncols=79):
        feats, labels = feats.to(device), labels.to(device)

        optimizer.zero_grad()

        pred = model(feats)
        loss = criterion(pred, labels)

        total_loss += loss.item()
        total_correct += pred.max(dim=-1)[1].eq(labels).sum().item()

        l2_weight = 1e-3
        n_weight, l2 = 0, 0.
        for name, param in model.named_parameters():
            if 'weight' in name:
                n_weight += 1
                l2 += param.pow(2).sum()
        loss += l2_weight * l2 / (2 * n_weight)

        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader), total_correct


def evaluate(model, dataloader, criterion):
    model.eval()

    total_loss, total_correct = 0, 0
    for feats, labels in tqdm(dataloader, leave=False, ncols=79):
        feats, labels = feats.to(device), labels.to(device)

        with torch.no_grad():
            pred = model(feats)
        loss = criterion(pred, labels)

        total_loss += loss.item()
        total_correct += pred.max(dim=-1)[1].eq(labels).sum().item()

    return total_loss / len(dataloader), total_correct


def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tblog = SummaryWriter(f"{args.model_dir}/log")

    # Define train_model
    if not args.attention:
        model = Xvector(args.feat_dim, args.num_spks, dropout=args.dropout_p)
    else:
        model = Xvector_AttnPooling(args.feat_dim, args.num_spks,
                                    dropout=args.dropout_p)
    model = model.to(device)
    model = nn.DataParallel(model)
    print(model)

    # Dataset
    egs_train, egs_valid = nnet3_egs_info(args.voxceleb_egs)

    # Loss function & Optimizer
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)

    gamma = math.exp(math.log(args.final_lr / args.initial_lr)
                     / (args.epochs * len(egs_train)))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    # Start training
    logger.info(f"Epochs: {args.epochs}, num_egs: {len(egs_train)}")
    for epoch in range(args.epochs):
        for n, egs in enumerate(egs_train):
            iteration = epoch * len(egs_train) + n + 1

            # Train step
            dataset = EgsDataset(egs)
            loader = DataLoader(dataset, batch_size=args.batch_size)

            train_loss, train_corr = train(model, loader, criterion,
                                           optimizer, iteration)
            train_acc = train_corr / len(dataset)
            dataset.file.close()

            # Evaluation step
            test_loss, test_acc, test_length = 0., 0., 0
            for egs in egs_valid:
                dataset = EgsDataset(egs)
                loader = DataLoader(dataset, batch_size=args.batch_size)
                test_length += len(dataset)

                loss, corr = evaluate(model, loader, criterion)
                test_loss += loss
                test_acc += corr
                dataset.file.close()
            test_loss = test_loss / len(egs_valid)
            test_acc = test_acc / test_length

            # Log
            logger.info(f"Iter {iteration}"
                        + f"|Train|loss:{train_loss:.4f}, acc:{train_acc:.4f}"
                        + f"|Test |loss:{test_loss:.4f}, acc:{test_acc:.4f}")
            tblog.add_scalars('Loss', {'Train': train_loss,
                                       'Test': test_loss}, iteration)
            tblog.add_scalars('Acc', {'Train': train_acc,
                                      'Test': test_acc}, iteration)
            scheduler.step()

        logger.info(f"Iter {iteration}| Save the final.pth ")
        state = model.module.state_dict()
        torch.save(state, f'{args.model_dir}/{epoch + 1}.pth')
        torch.save(state, f'{args.model_dir}/final.pth')
    tblog.close()


if __name__ == "__main__":
    main()
