import time
from pathlib import Path

import torch
torch.manual_seed(4899)
from torch.utils.data import DataLoader

from Ds.piqa_ds import PiqaDataset
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
    input_format = args.input_format

    model = Model(
        name=name,
        num_choices=args.num_choices
    ).cuda()

    sep_token = model.tokenizer.sep_token

    optimizer = configure_optimizer(model, args)

    json_path_train = "/content/DNLP_project/data/clarified_piqa/train.jsonl"

    train_dataset = PiqaDataset(
                        json_path_train, 
                        sep_token=sep_token, 
                        input_format=input_format, 
                        shuffle=True
                        )

    
    train_loader = DataLoader(
                        train_dataset, 
                        shuffle=shuffle, 
                        batch_size=train_batch_size,
                        collate_fn=train_dataset.collate_fn
                        )

    dev_dataloader_list = create_dev_dl(model)

    Path("/content/DNLP_project/log/piqa_clarified").mkdir(parents=True, exist_ok=True)
    lf_name = "/content/DNLP_project/log/piqa_clarified" + name.replace("/", "-") + ".txt"
    lf = open(lf_name, "a")
    lf.write(str(args) + "\n\n")
    lf.close()
    
    start_time = time.time()
    for e in range(epochs):

        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer)
       
        x = "Epoch {}: Loss: Train {};".format(e + 1, train_loss)
        y1 = "Classification Acc: Train {};".format(train_acc)
        y2 = "Classification Macro F1: Train {};".format(train_f1)
        
        print(x)
        print(y1)
        print(y2)

        lf = open(lf_name, "a")
        lf.write(x + "\n" + y1 + "\n" + y2 + "\n" + "\n\n")
        lf.close()
    
    training_time = time.time() - start_time
    print('Training time:', training_time )
    
    lf = open(lf_name, "a")
    lf.write('Training time: {}'.format(training_time) + "\n")
    lf.close()

    acc_list = list()

    lf = open(lf_name, "a")
    for dev_loader in dev_dataloader_list:
        a = "Knowledge source: {}".format(dev_loader[0])
        print(a)
        lf.write(a + "\n")
        preds, ins_acc  = test(model, dev_loader[1])
        b = "Instance accuracy: {}".format(ins_acc)
        print(b)
        lf.write(b + "\n")
        acc_list.append(ins_acc)
    avg = sum(acc_list)/len(acc_list)
    print("Average, across knowledge sources, Instance Accuracy: {}".format(avg))
    lf.write("Average, across knowledge sources, Instance Accuracy: {}".format(avg) + "\n")
    lf.write("-" * 100 + "\n")
    lf.close()

