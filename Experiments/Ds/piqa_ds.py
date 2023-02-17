import json
import random
random.seed(129864)
import pandas as pd

from torch.utils.data import Dataset

from Utils.custom_parser import my_parser

'''
    Class to handle the PIQA clarified, without clarifications (only train split)
'''
class PiqaDataset(Dataset):
    '''
        f = path to json file
        shuffle = True/False to shuffle input data or not
    '''
    def __init__(self, f, shuffle):

        ''' 
            content : list() = all the possible combinations (question, answer)
            labels : list() = list with 0 and 1, 1 where we have the correct answer
        '''
        content, labels = [], []
        x = open(f).readlines()
        
        if shuffle:
            random.shuffle(x)

        for line in x:

            '''
                create a python dict from this line of the jsonl
            '''
            instance = json.loads(line)

            '''
                take question
            '''
            question = instance["goal"]
            '''
                take possible answers
            '''
            a1, a2 = instance["sol1"], instance["sol2"]
            '''
                take label of correct answer (0 or 1)
            '''
            l = instance["label"]

            '''
                create content list
            '''
            content.append("{} {}".format(question, a1))
            content.append("{} {}".format(question, a2))
            
            '''
                create labels list
            '''
            if l == 0:
                labels += [1, 0]
            elif l == 1:
                labels += [0, 1]
                
        self.content, self.labels = content, labels
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index):
        s1, s2 = self.content[index], self.labels[index]
        return s1, s2
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]

'''
    class to handle PIQA clarified dataset (validation and test splits)
'''
class ClarifiedPiqaDataset(Dataset):
    '''
        very similar structure as PiqaDataset class
        here we save only clarifications
        and we create the content list adding them
        all values in content tuples are separeted by sep_token taken from model.tokenizer
    '''
    def __init__(self, f, shuffle, sep_token):

        args = my_parser()

        content, labels = [], []
        x = open(f).readlines()
        
        if shuffle:
            random.shuffle(x)

        for line in x:
            instance = json.loads(line)

            question = instance["goal"]
            a1, a2 = instance["sol1"], instance["sol2"]
            '''
                because in test split we do not have correct answers labels, so we create a dummy one
                this will be not used since we only perform prediction in test and no accuracy
            '''
            if "label" in instance:
                l = instance["label"]
            else:
                l = 0

            clarifications = [c[1] if len(c[1].split()) > 1 else " ".join((c)) for c in instance['clarifications']]
            clarifications = [c[0].upper() + c[1:] for c in clarifications] + [""]
            
            
            if len(clarifications) >= args.max_clarifications:
                clarifications = random.sample(clarifications, args.max_clarifications)
                
            content.append("{} {} {} {} {}".format(question, sep_token, a1, sep_token, clarifications))
            content.append("{} {} {} {} {}".format(question, sep_token, a2, sep_token, clarifications))

            if l == 0:
                labels += [1, 0]
            elif l == 1:
                labels += [0, 1]
                
        self.content, self.labels = content, labels
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index):
        s1, s2 = self.content[index], self.labels[index]
        return s1, s2
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]

            
