import math
import numpy as np
import pandas as pd
from tqdm import tqdm 
import torch
torch.manual_seed(4899)
from torch.optim import AdamW

from transformers.optimization import get_scheduler

from sklearn.metrics import accuracy_score, f1_score

from Utils.custom_parser import my_parser

args = my_parser()

def configure_optimizer(model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.wd,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)

    return optimizer


def configure_scheduler(optimizer, num_training_steps, args):
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps > 0
        else math.ceil(num_training_steps * args.warmup_ratio)
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return lr_scheduler

def train(model, dataloader, optimizer=None):

    losses, preds, preds_cls, labels_cls, = [], [], [], []
    model.train()

    for batch in tqdm(dataloader, leave=False):
        optimizer.zero_grad()

        content, l_cls = batch
        loss, p, p_cls = model(batch)

        preds.append(p)
        preds_cls.append(p_cls)
        labels_cls.append(l_cls)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    avg_loss = round(np.mean(losses), 4)

    all_preds_cls = [item for sublist in preds_cls for item in sublist]
    all_labels_cls = [item for sublist in labels_cls for item in sublist]
    acc = round(accuracy_score(all_labels_cls, all_preds_cls), 4)
    f1 = round(f1_score(all_labels_cls, all_preds_cls, average="macro"), 4)

    return avg_loss, acc, f1

def eval(model, dataloader, optimizer=None):

    losses, preds, preds_cls, labels_cls, = [], [], [], []
    model.eval()

    for batch in tqdm(dataloader, leave=False):
        content, l_cls = batch
        loss, p, p_cls = model(batch)

        preds.append(p)
        preds_cls.append(p_cls)
        labels_cls.append(l_cls)

        losses.append(loss.item())
    
    avg_loss = round(np.mean(losses), 4)

    all_preds_cls = [item for sublist in preds_cls for item in sublist]
    all_labels_cls = [item for sublist in labels_cls for item in sublist]
    acc = round(accuracy_score(all_labels_cls, all_preds_cls), 4)
    f1 = round(f1_score(all_labels_cls, all_preds_cls, average="macro"), 4)

    instance_preds = [item for sublist in preds for item in sublist]
    instance_labels = np.array(all_labels_cls).reshape(-1, args.num_choices).argmax(1)
    instance_acc = round(accuracy_score(instance_labels, instance_preds), 4)

    return avg_loss, acc, instance_acc, f1

def test(model, dataloader, optimizer=None):

    losses, preds, preds_cls, labels_cls, = [], [], [], []
    model.eval()

    for batch in tqdm(dataloader, leave=False):

        content, l_cls = batch
        loss, p, p_cls = model(batch)

        preds.append(p)
        preds_cls.append(p_cls)
        labels_cls.append(l_cls)

        losses.append(loss.item())

    avg_loss = round(np.mean(losses), 4)

    all_labels_cls = [item for sublist in labels_cls for item in sublist]
    
    instance_preds = [item for sublist in preds for item in sublist]
    instance_labels = np.array(all_labels_cls).reshape(-1, args.num_choices).argmax(1)
    instance_acc = round(accuracy_score(instance_labels, instance_preds), 4)
    print("Test Instance Accuracy :", instance_acc)

    mapper = {0: "1", 1: "2", 2: "3", 3: "4"}
    instance_preds = [mapper[item] for item in instance_preds]
    print("Test preds frequency:", dict(pd.Series(instance_preds).value_counts()))

    return instance_preds, instance_acc

