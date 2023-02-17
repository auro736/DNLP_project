import os

from Ds.piqa_ds import ClarifiedPiqaDataset
from Utils.custom_parser import my_parser

from torch.utils.data import DataLoader
import torch
torch.manual_seed(4899)

from Models.model import Model


'''
    Function used to create Dataloaders from a folder of validation files, then it stores them in a list and return it
'''
def create_dev_dl(model):

    args = my_parser()

    dev_folder = "/content/DNLP_project/data/clarified_piqa/dev/"

    dev_dataloader_list = list()

    with os.scandir(dev_folder) as entries:
        for entry in entries:
            full_path = os.path.join(dev_folder,entry)
            
            dev_dataset = ClarifiedPiqaDataset(
                            full_path, 
                            sep_token=model.tokenizer.sep_token, 
                            shuffle=False
                        )
            
            dev_dataloader = DataLoader(
                                dev_dataset, 
                                shuffle=False, 
                                batch_size=args.eval_bs, 
                                collate_fn=dev_dataset.collate_fn
                                )
            
            dev_dataloader_list.append((entry,dev_dataloader))

    #print(len(dev_dataloader_list))
    return dev_dataloader_list

