import os

from Ds.piqa_ds import ClarifiedPiqaDataset
from Utils.custom_parser import my_parser

from torch.utils.data import DataLoader
import torch
torch.manual_seed(4899)

from Models.model import Model

def create_dev_dl(model):

    args = my_parser()

    dev_folder = "/content/DNLP_project/data/clarified_piqa/dev/"
    #dev_folder = "/mnt/c/Users/auror/Desktop/NUOVA REPO PROGETTO DNLP/DNLP_project/data/clarified_piqa/dev"

    dev_dataloader_list = list()

    with os.scandir(dev_folder) as entries:
        for entry in entries:
            full_path = os.path.join(dev_folder,entry)
            #print(full_path)
            
            dev_dataset = ClarifiedPiqaDataset(
                            full_path, 
                            sep_token=model.tokenizer.sep_token, 
                            input_format=args.input_format, 
                            shuffle=False
                        )
            
            dev_dataloader = DataLoader(
                                dev_dataset, 
                                shuffle=False, 
                                batch_size=args.eval_bs, 
                                collate_fn=dev_dataset.collate_fn
                                )
            
            dev_dataloader_list.append((entry,dev_dataloader))

    print(len(dev_dataloader_list))
    return dev_dataloader_list

# if __name__ == '__main__':
#     args = my_parser()
#     print(args)

#     train_batch_size = args.bs
#     eval_batch_size = args.eval_bs
#     test_batch_size = args.eval_bs
#     epochs = args.epochs
#     name = args.name
#     shuffle = args.shuffle
#     input_format = args.input_format

#     # num_choices = 2
#     # vars(args)["num_choices"] = num_choices
#     # assert eval_batch_size % num_choices == 0, "Eval batch size should be a multiple of num choices, which is 4"

#     model = Model(
#         name=name,
#         num_choices=args.num_choices
#     ).cuda() 

#     create_dev_dl(model)
