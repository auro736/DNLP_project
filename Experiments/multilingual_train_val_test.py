import time
import math
import numpy as np
from os import path
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


# from transformers import get_linear_schedule_with_warmup
# from transformers.trainer_pt_utils import get_parameter_names
from transformers.optimization import get_scheduler

from sklearn.metrics import accuracy_score, f1_score

from Utils.multilingual_ds import MultilingualDataset
from Utils.custom_parser import my_parser

# from model import Model
from Models.model import Model


# def configure_dataloaders(json_path_train, json_path_valid, json_path_test, train_batch_size=4, eval_batch_size=4, test_batch_size=4, shuffle=False, sep_token=None, input_format=0):
    
#     json_path_train = "/content/TEAM/data/persian/train.jsonl"
#     json_path_valid = "/content/TEAM/data/persian/valid.jsonl"
#     json_path_test = "/content/TEAM/data/persian/test.jsonl"

#     train_dataset = PersianDataset(json_path_train, sep_token=sep_token, input_format=input_format, shuffle=True)
#     train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=train_batch_size,
#                               collate_fn=train_dataset.collate_fn)

#     val_dataset = PersianDataset(json_path_valid, sep_token=sep_token, input_format=input_format, shuffle=False)
#     val_loader = DataLoader(val_dataset, shuffle=False, batch_size=eval_batch_size, collate_fn=val_dataset.collate_fn)

#     test_dataset = PersianDataset(json_path_test, sep_token=sep_token, input_format=input_format, shuffle=False)
#     test_loader = DataLoader(test_dataset, shuffle=False, batch_size=test_batch_size, collate_fn=test_dataset.collate_fn)

#     return train_loader, val_loader, test_loader


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


def train_or_eval_model(model, dataloader, optimizer=None, split="Train"):

    losses, preds, preds_cls, labels_cls, = [], [], [], []

    if split == "Train":
        model.train()
    else:
        model.eval()

    for batch in tqdm(dataloader, leave=False):
        if split == "Train":
            optimizer.zero_grad()

        content, l_cls = batch
        loss, p, p_cls = model(batch)

        preds.append(p)
        preds_cls.append(p_cls)
        labels_cls.append(l_cls)

        if split == "Train":
            loss.backward()
            optimizer.step()

        losses.append(loss.item())

    avg_loss = round(np.mean(losses), 4)

    if split == "Train":

        all_preds_cls = [item for sublist in preds_cls for item in sublist]
        all_labels_cls = [item for sublist in labels_cls for item in sublist]
        acc = round(accuracy_score(all_labels_cls, all_preds_cls), 4)
        f1 = round(f1_score(all_labels_cls, all_preds_cls, average="macro"), 4)

        return avg_loss, acc, f1

    elif split == "Val":

        all_preds_cls = [item for sublist in preds_cls for item in sublist]
        all_labels_cls = [item for sublist in labels_cls for item in sublist]
        acc = round(accuracy_score(all_labels_cls, all_preds_cls), 4)
        f1 = round(f1_score(all_labels_cls, all_preds_cls, average="macro"), 4)

        instance_preds = [item for sublist in preds for item in sublist]
        instance_labels = np.array(all_labels_cls).reshape(-1, args.num_choices).argmax(1)
        instance_acc = round(accuracy_score(instance_labels, instance_preds), 4)

        return avg_loss, acc, instance_acc, f1

    elif "Test" in split:

        all_labels_cls = [item for sublist in labels_cls for item in sublist]

       
        instance_preds = [item for sublist in preds for item in sublist]
        instance_labels = np.array(all_labels_cls).reshape(-1, args.num_choices).argmax(1)
        instance_acc = round(accuracy_score(instance_labels, instance_preds), 4)
        print("Test Instance Accuracy :", instance_acc)

        mapper = {0: "A", 1: "B", 2: "C", 3: "D"}
        instance_preds = [mapper[item] for item in instance_preds]
        print("Test preds frequency:", dict(pd.Series(instance_preds).value_counts()))

        return instance_preds

def avg(list):

    return round(sum(list)/len(list),4)

if __name__ == "__main__":

    args = my_parser()
    print(args)

    train_batch_size = args.bs
    eval_batch_size = args.eval_bs
    test_batch_size = args.eval_bs
    epochs = args.epochs
    name = args.name
    shuffle = args.shuffle
    input_format = args.input_format

    num_choices = 4
    vars(args)["num_choices"] = num_choices
    assert eval_batch_size % num_choices == 0, "Eval batch size should be a multiple of num choices, which is 4"

    model = Model(
        name=name,
        num_choices=num_choices
    ).cuda()

    model_ckp_path = f"/content/DNLP_project/Experiments/Checkpoints/{name}.pth"

    if path.exists(model_ckp_path):
        model.load_state_dict(torch.load(model_ckp_path))

    sep_token = model.tokenizer.sep_token

    opt_ckp_path = f"/content/DNLP_project/Experiments/Checkpoints/{name}_optimizer.pth"

    optimizer = configure_optimizer(model, args)

    if path.exists(opt_ckp_path):
        optimizer.load_state_dict(torch.load(opt_ckp_path))

    # json_path_train = "/content/DNLP_project/data/multilingual/train.jsonl"
    # json_path_valid = "/content/DNLP_project/data/multilingual/dev.jsonl"
    # json_path_test = "/content/DNLP_project/data/multilingual/test.jsonl"

    '''
        Prova con test e train files invertiti
    '''
    json_path_train = "/content/DNLP_project/data/multilingual/new_test.jsonl"
    json_path_valid = "/content/DNLP_project/data/multilingual/dev.jsonl"
    json_path_test = "/content/DNLP_project/data/multilingual/train.jsonl"

    # json_path_train = "/mnt/c/Users/auror/Desktop/NUOVA REPO PROGETTO DNLP/DNLP_project/data/multilingual/train.jsonl"
    # json_path_valid = "/mnt/c/Users/auror/Desktop/NUOVA REPO PROGETTO DNLP/DNLP_project/data/multilingual/dev.jsonl"
    # json_path_test = "/mnt/c/Users/auror/Desktop/NUOVA REPO PROGETTO DNLP/DNLP_project/data/multilingual/test.jsonl"

    train_dataset = MultilingualDataset(
                        json_path_train, 
                        sep_token=sep_token, 
                        input_format=input_format, 
                        shuffle=True
                        )
    
    val_dataset = MultilingualDataset(
                        json_path_valid, 
                        sep_token=sep_token, 
                        input_format=input_format, 
                        shuffle=False
                        )

    test_dataset = MultilingualDataset(
                        json_path_test, 
                        sep_token=sep_token, 
                        input_format=input_format, 
                        shuffle=False
                        )
    
    train_loader = DataLoader(
                        train_dataset, 
                        shuffle=shuffle, 
                        batch_size=train_batch_size,
                        collate_fn=train_dataset.collate_fn
                        )

    val_loader = DataLoader(
                        val_dataset, 
                        shuffle=False, 
                        batch_size=eval_batch_size, 
                        collate_fn=val_dataset.collate_fn
                        )

    test_loader = DataLoader(
                        test_dataset, 
                        shuffle=False, 
                        batch_size=test_batch_size, 
                        collate_fn=test_dataset.collate_fn
                        )


    # if "/" in name:
    #     sp = name[name.index("/") + 1:]
    # else:
    #     sp = name

    exp_id = str(int(time.time()))
    vars(args)["exp_id"] = exp_id
    rs = "Acc: {}"

    path = "/content/DNLP_project/saved/multilingual_dataset/" + exp_id + "/" + name.replace("/", "-")
    Path("/content/DNLP_project/saved/multilingual_dataset/" + exp_id + "/").mkdir(parents=True, exist_ok=True)

    fname = "/content/DNLP_project/saved/multilingual_dataset/" + exp_id + "/" + "args.txt"

    f = open(fname, "a")
    f.write(str(args) + "\n\n")
    f.close()

    Path("/content/DNLP_project/results/multilingual_dataset/").mkdir(parents=True, exist_ok=True)
    lf_name = "/content/DNLP_project/results/multilingual_dataset/" + name.replace("/", "-") + ".txt"
    lf = open(lf_name, "a")
    lf.write(str(args) + "\n\n")
    lf.close()

    # path = "saved/multilingual_dataset/" + exp_id + "/" + name.replace("/", "-")
    # Path("saved/multilingual_dataset/" + exp_id + "/").mkdir(parents=True, exist_ok=True)

    # fname = "saved/multilingual_dataset/" + exp_id + "/" + "args.txt"

    # f = open(fname, "a")
    # f.write(str(args) + "\n\n")
    # f.close()

    # Path("results/multilingual_dataset/").mkdir(parents=True, exist_ok=True)
    # lf_name = "results/multilingual_dataset/" + name.replace("/", "-") + ".txt"
    # lf = open(lf_name, "a")
    # lf.write(str(args) + "\n\n")
    # lf.close()

    val_ins_acc_list = list()

    for e in range(epochs):

        #torch.cuda.empty_cache()

        # train_loader, val_loader, test_loader = configure_dataloaders(json_path_train, json_path_valid, json_path_test,
        #                                                                 train_batch_size, eval_batch_size, test_batch_size, shuffle,
        #                                                               sep_token=sep_token, input_format=input_format)
        
        train_loss, train_acc, train_f1 = train_or_eval_model(model, train_loader, optimizer, split = "Train")
        val_loss, val_acc, val_ins_acc, val_f1 = train_or_eval_model(model, val_loader, split="Val")
        # test_preds = train_or_eval_model(model, test_loader, split="Test")

        val_ins_acc_list.append(val_ins_acc)

        # with open(path + "-epoch-" + str(e + 1) + ".txt", "w") as f:
        #     f.write("\n".join(list(test_preds)))

        x = "Epoch {}: Loss: Train {}; Val {}".format(e + 1, train_loss, val_loss)
        y1 = "Classification Acc: Train {}; Val {}".format(train_acc, val_acc)
        y2 = "Classification Macro F1: Train {}; Val {}".format(train_f1, val_f1)
        z = "Instance Acc: Val {}".format(val_ins_acc)

        print(x)
        print(y1)
        print(y2)
        print(z)

        lf = open(lf_name, "a")
        lf.write(x + "\n" + y1 + "\n" + y2 + "\n" + z + "\n\n")
        lf.close()

        f = open(fname, "a")
        f.write(x + "\n" + y1 + "\n" + y2 + "\n" + z + "\n\n")
        f.close()

    torch.save(model.state_dict(), model_ckp_path)
    torch.save(optimizer.state_dict(), opt_ckp_path)
    
    avg_ins_acc = f"Average Instance Accuracy Val: {avg(val_ins_acc_list)}"
    print(avg_ins_acc)

    test_preds = train_or_eval_model(model, test_loader, split="Test")

    with open(path + "-epoch-" + str(e + 1) + ".txt", "w") as f:
            f.write("\n".join(list(test_preds)))

    lf = open(lf_name, "a")
    lf.write("-" * 100 + "\n")
    lf.write(avg_ins_acc + "\n")
    lf.close()