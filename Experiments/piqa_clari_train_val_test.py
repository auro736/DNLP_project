import time
from pathlib import Path

import torch
torch.manual_seed(4899)
from torch.utils.data import DataLoader

from Ds.piqa_ds import PiqaDataset, ClarifiedPiqaDataset
from Utils.custom_parser import my_parser

from Models.model import Model

from Utils.utils import *

from prepare_piqa_clarifications import create_dev_dl

if __name__ == "__main__":

    args = my_parser()
    print(args)

    train_batch_size = args.bs
    eval_batch_size = args.eval_bs
    test_batch_size = args.eval_bs
    epochs = args.epochs
    name = args.name
    shuffle = args.shuffle

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
    json_path_train = "/content/DNLP_project/data/clarified_piqa/train.jsonl"
    json_path_test = "/content/DNLP_project/data/clarified_piqa/test_clarified_xlnet-base-cased.jsonl"

    '''
        Configure Dataset objencts and Dataloaders
    '''
    train_dataset = PiqaDataset(
                        json_path_train, 
                        shuffle=True
                        )

    
    train_loader = DataLoader(
                        train_dataset, 
                        shuffle=shuffle, 
                        batch_size=train_batch_size,
                        collate_fn=train_dataset.collate_fn
                        )

    '''
        For validation set, since we have 11 different files, use the function create_dev_dl, it return a list of dataloaders 
    '''
    dev_dataloader_list = create_dev_dl(model)


    test_dataset = ClarifiedPiqaDataset(
                        json_path_test, 
                        sep_token=sep_token, 
                        shuffle=False
                        )

    
    test_loader = DataLoader(
                        test_dataset, 
                        shuffle=shuffle, 
                        batch_size=train_batch_size,
                        collate_fn=train_dataset.collate_fn
                        )


    Path("/content/DNLP_project/log/piqa_clarified/").mkdir(parents=True, exist_ok=True)
    lf_name = "/content/DNLP_project/log/piqa_clarified/" + name.replace("/", "-") + ".txt"
    lf = open(lf_name, "a")
    lf.write(str(args) + "\n\n")
    lf.close()

    start_time = time.time()
    for e in range(epochs):

        acc_list = list()
        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer)

        # dev_loader is a tuple of (name of LM, dataloader related)
        for dev_loader in dev_dataloader_list: 
            val_loss, val_acc, val_ins_acc, val_f1  = eval(model, dev_loader[1])
            acc_list.append(val_ins_acc)

        avg = round(sum(acc_list)/len(acc_list),4)
        
        x = "Epoch {}: Loss: Train {}; Val {}".format(e + 1, train_loss, val_loss)
        z = "Average, across knowledge sources, Instance Accuracy: Val {}".format(avg)

        print(x)
        print(z)

        lf = open(lf_name, "a")
        lf.write(x + "\n" + z + "\n\n")
        lf.close()
    
    training_time = time.time() - start_time
    print('Training time:', training_time )
    
    lf = open(lf_name, "a")
    lf.write('Training time: {}'.format(training_time) + "\n")
    lf.close()

    Path("/content/DNLP_project/log/piqa_clarified/predictions/").mkdir(parents=True, exist_ok=True)
    path_pred = "/content/DNLP_project/log/piqa_clarified/predictions/" + name.replace("/", "-") + "_preds.txt"

    print("Making test predictions...")
    start_time = time.time()
    test_preds = test(model, test_loader, ds = 'piqa_clarified')
    print('Execution time:', time.time() - start_time)

    with open(path_pred,"a") as f:
        f.write(str(args) + "\n\n")
        f.write("\n".join(list(test_preds)))

    lf = open(lf_name, "a")
    lf.write("-" * 100 + "\n")
    lf.close()

