import time
from pathlib import Path

import torch
torch.manual_seed(4899)
from torch.utils.data import DataLoader

from Ds.persian_ds import PersianDataset

from Utils.custom_parser import my_parser
from Utils.utils import *

from Models.model import Model


if __name__ == "__main__":

    args = my_parser()
    print(args)

    train_batch_size = args.bs
    eval_batch_size = args.eval_bs
    test_batch_size = args.eval_bs
    epochs = args.epochs
    name = args.name
    shuffle = args.shuffle

    assert eval_batch_size % args.num_choices == 0, "Eval batch size should be a multiple of num choices, which is 4"
    
    '''
        Model instantiation
    '''
    model = Model(
        name=name,
        num_choices=args.num_choices
    ).cuda()

    sep_token = model.tokenizer.sep_token

    optimizer = configure_optimizer(model, args)

    '''
        Input paths
    '''
    json_path_train = "/content/DNLP_project/data/persian/train.jsonl"
    json_path_valid = "/content/DNLP_project/data/persian/valid.jsonl"
    json_path_test_lit = "/content/DNLP_project/data/persian/test_lit.jsonl"
    json_path_test_ck = "/content/DNLP_project/data/persian/test_ck.jsonl"
    json_path_test_ml = "/content/DNLP_project/data/persian/test_ml.jsonl"

    '''
        Configure Dataset objencts and Dataloaders
    '''
    train_dataset = PersianDataset(
                        json_path_train, 
                        shuffle=True
                        )
    
    val_dataset = PersianDataset(
                        json_path_valid, 
                        shuffle=False
                        )

    test_dataset_lit = PersianDataset(
                        json_path_test_lit, 
                        shuffle=False
                        )
    
    test_dataset_ck = PersianDataset(
                        json_path_test_ck, 
                        shuffle=False
                        )
    
    test_dataset_ml = PersianDataset(
                        json_path_test_ml, 
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

    test_loader_lit = DataLoader(
                        test_dataset_lit, 
                        shuffle=False, 
                        batch_size=test_batch_size, 
                        collate_fn=test_dataset_lit.collate_fn
                        )

    test_loader_ck = DataLoader(
                        test_dataset_ck, 
                        shuffle=False, 
                        batch_size=test_batch_size, 
                        collate_fn=test_dataset_ck.collate_fn
                        )

    test_loader_ml = DataLoader(
                        test_dataset_ml, 
                        shuffle=False, 
                        batch_size=test_batch_size, 
                        collate_fn=test_dataset_ml.collate_fn
                        )                    


    Path("/content/DNLP_project/log/persian/").mkdir(parents=True, exist_ok=True)
    lf_name = "/content/DNLP_project/log/persian/" + name.replace("/", "-") + ".txt"
    lf = open(lf_name, "a")
    lf.write(str(args) + "\n\n")
    lf.close()

    print("Start training....")
    start_time = time.time()
    for e in range(epochs):
        
        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer)
        val_loss, val_acc, val_ins_acc, val_f1 = eval(model, val_loader)

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

    training_time = time.time() - start_time
    print('Training time:', training_time )
    lf = open(lf_name, "a")
    lf.write('Training time: {}'.format(training_time) + "\n")
    lf.close()

    print("Testing...")

    Path("/content/DNLP_project/log/persian/predictions/").mkdir(parents=True, exist_ok=True)
    path_lit_pred = "/content/DNLP_project/log/persian/predictions/" + name.replace("/", "-") + "_lit_pred.txt"
    path_ck_pred = "/content/DNLP_project/log/persian/predictions/" + name.replace("/", "-") + "_ck_pred.txt"
    path_ml_pred = "/content/DNLP_project/log/persian/predictions/" + name.replace("/", "-") + "_ml_pred.txt"

    print("Results for test LIT")
    start_time = time.time()
    test_preds_lit, ins_acc_lit = test(model, test_loader_lit, ds = 'persian')
    print('Execution time:', time.time() - start_time)

    with open(path_lit_pred,"a") as f:
        f.write(str(args) + "\n\n")
        f.write("\n".join(list(test_preds_lit)))

    print("Results for test CK")
    start_time = time.time()
    test_preds_ck, ins_acc_ck = test(model, test_loader_ck, ds = 'persian')
    print('Execution time:', time.time() - start_time)

    with open(path_ck_pred,"a") as f:
        f.write(str(args) + "\n\n")
        f.write("\n".join(list(test_preds_ck)))

    print("Results for test ML")
    start_time = time.time()
    test_preds_ml, ins_acc_ml  = test(model, test_loader_ml, ds = 'persian')
    print('Execution time:', time.time() - start_time)

    with open(path_ml_pred,"a") as f:
        f.write(str(args) + "\n\n")
        f.write("\n".join(list(test_preds_ml)))

    lf = open(lf_name, "a")
    lf.write("Instance Acc: Test LIT {}".format(ins_acc_lit)+"\n")
    lf.write("Instance Acc: Test CK {}".format(ins_acc_ck)+"\n")
    lf.write("Instance Acc: Test ML {}".format(ins_acc_ml)+"\n")
    lf.write("-" * 100 + "\n")
    lf.close()