import json
import random
random.seed(129864)
import pandas as pd

from torch.utils.data import Dataset

from Utils.custom_parser import my_parser

'''
    Class to handle the persian dataset
'''
class PersianDataset(Dataset):
    '''
        f = path to json file
        shuffle = True/False to shuffle input data or not
    '''
    def __init__(self, f, shuffle):

        args = my_parser()

        ''' 
            content : list() = all the possible combinations (question, answer) or (question, answer, category)
            labels : list() = list with 0 and 1, 1 where we have the correct answer
        '''
        content, labels = [], []
        x = open(f).readlines()
        if shuffle:
            random.shuffle(x)
        
        self.corr_ans_ids = list()

        for line in x:
            '''
                create a python dict from this line of the jsonl
            '''
            instance = json.loads(line)

            ''' 
                possible answers length may vary from 3 to 5 only one is correct
                for network purposes we need to have the number of choices fixed
                aribitrary chosen as 4, so we proceed only if len(choices) == 4
            '''

            if len(instance["candidates"]) == 4:

                '''
                    take question
                '''
                question = instance["question"]
                '''
                    id (increasing number from 1 to 4) of the correct answer
                '''
                correct_answer_id = int(instance["answer"])

                '''
                    save also question category
                '''
                category = instance["category"]

                '''
                    save all given answers
                '''
                a1 = instance["candidates"][0]
                a2 = instance["candidates"][1]
                a3 = instance["candidates"][2]
                a4 = instance["candidates"][3]
                
                '''
                    create content list 
                '''
                if args.use_categories == 1:
                    content.append("{} {} {}".format(question, a1, category))
                    content.append("{} {} {}".format(question, a2, category))
                    content.append("{} {} {}".format(question, a3, category))
                    content.append("{} {} {}".format(question, a4, category))
                elif args.use_categories == 0:
                    content.append("{} {}".format(question, a1))
                    content.append("{} {}".format(question, a2))
                    content.append("{} {}".format(question, a3))
                    content.append("{} {}".format(question, a4))

                '''
                    create labels list
                '''
                if correct_answer_id == 1:
                    labels += [1, 0, 0, 0]
                elif correct_answer_id == 2:
                    labels += [0, 1, 0, 0]
                elif correct_answer_id == 3:
                    labels += [0, 0, 1, 0]
                elif correct_answer_id == 4:
                    labels += [0, 0, 0, 1]
                
        self.content, self.labels = content, labels
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index):
        s1, s2 = self.content[index], self.labels[index]
        return s1, s2
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
            
