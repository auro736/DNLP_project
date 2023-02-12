import json
import random
random.seed(129864)
import pandas as pd

from torch.utils.data import Dataset

from Utils.custom_parser import my_parser

class PiqaDataset(Dataset):

    def __init__(self, f, shuffle):

        '''
        f = path to json file
        sep_token = token separatore(?)
        input_format = perchè i vari json sono organizzati in modo diverso, capire come fare questo
        shuffle = shuffle del dataloader
        '''

        ''' content : list() = all the possible combinations (question, answer)
            labels : list() = list with 0 and 1, 1 where we have the correct answer'''

        content, labels = [], []
        x = open(f).readlines()
        
        if shuffle:
            random.shuffle(x)

        for line in x:

            '''create a python dict from this line of the jsonl'''
            instance = json.loads(line)

            question = instance["goal"]
            a1, a2 = instance["sol1"], instance["sol2"]
            l = instance["label"]
            '''create all the possible combinations (question, answer)'''

            # if input_format == "0":
            content.append("{} {}".format(question, a1))
            content.append("{} {}".format(question, a2))
            # elif input_format == "1":
            #     content.append("{} {} {}".format(question, sep_token, a1))
            #     content.append("{} {} {}".format(question, sep_token, a2))
            # elif input_format == "2":
            #     content.append("goal: {} {} solution: {}".format(question, sep_token, a1))
            #     content.append("goal: {} {} solution: {}".format(question, sep_token, a2))
            
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


class ClarifiedPiqaDataset(Dataset):

    def __init__(self, f, shuffle, sep_token):

        args = my_parser()

        '''
        f = path to json file
        sep_token = token separatore(?)
        input_format = perchè i vari json sono organizzati in modo diverso, capire come fare questo
        shuffle = shuffle del dataloader
        '''

        ''' content : list() = all the possible combinations (question, answer)
            labels : list() = list with 0 and 1, 1 where we have the correct answer'''

        content, labels = [], []
        x = open(f).readlines()
        
        if shuffle:
            random.shuffle(x)

        for line in x:

            '''create a python dict from this line of the jsonl'''
            instance = json.loads(line)

            question = instance["goal"]
            a1, a2 = instance["sol1"], instance["sol2"]
            if "label" in instance:
                l = instance["label"]
            else:
                l = 0
            '''create all the possible combinations (question, answer)'''

            clarifications = [c[1] if len(c[1].split()) > 1 else " ".join((c)) for c in instance['clarifications']]
            clarifications = [c[0].upper() + c[1:] for c in clarifications] + [""]
            
            
            if len(clarifications) >= args.max_clarifications:
                clarifications = random.sample(clarifications, args.max_clarifications)

            # if input_format == "0":
            content.append("{} {} {} {} {}".format(question, sep_token, a1, sep_token, clarifications))
            content.append("{} {} {} {} {}".format(question, sep_token, a2, sep_token, clarifications))
            # elif input_format == "1":
            #         content.append("goal: {} {} solution: {} {} clarifications: {}".format(question, sep_token, a1, sep_token, clarifications))
            #         content.append("goal: {} {} solution: {} {} clarifications: {}".format(question, sep_token, a2, sep_token, clarifications))

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

            
