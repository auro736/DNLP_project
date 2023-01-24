import json
import random
random.seed(129864)
import pandas as pd

from torch.utils.data import Dataset

class FrenchQascDataset(Dataset):

    def __init__(self, f, shuffle,sep_token,input_format):
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

            question = instance["question"]["stem"]

            ''' possible answers, length may vary from 3 to 5 only one is correct
                for network purposes we  need to have the number of choices fixed
                aribitrary chosen as 4, so we proceed only if len(choices) == 4
            '''
            choices = [ a["text"] for a in instance["question"]["choices"] ]

            l = instance["answerKey"]

            '''create all the possible combinations (question, answer)'''

            if input_format == "0":
                for c in choices:
                    content.append("{} {}".format(question, c))
            elif input_format == "1":
                for c in choices:
                    content.append("{} \\n {}".format(question, c))
            elif input_format == "2":
                for c in choices:
                    content.append("{} {} {}".format(question, sep_token, c))
                
            answers = ["A", "B", "C", "D", "E", "F", "G", "H"]
            y = [0, 0, 0, 0, 0, 0, 0, 0]
            y[answers.index(l)] = 1
            labels += y
                
        self.content, self.labels = content, labels
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index):
        s1, s2 = self.content[index], self.labels[index]
        return s1, s2
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
            
