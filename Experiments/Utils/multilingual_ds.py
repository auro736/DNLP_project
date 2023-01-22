import json
import random
import pandas as pd

from torch.utils.data import Dataset

class MultilingualDataset(Dataset):
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
            print("Dataset shuffled")
            random.shuffle(x)
        
        #self.corr_ans_ids = list()

        # c_3 = 0
        # c_4 = 0
        # c_5 = 0
        # error = 0

        for line in x:

            '''create a python dict from this line of the jsonl'''
            instance = json.loads(line)

            ''' possible answers, length may vary from 3 to 5 only one is correct
                for network purposes we  need to have the number of choices fixed
                aribitrary chosen as 4, so we proceed only if len(choices) == 4
            '''
            choices = [ a["text"] for a in instance["question"]["choices"] ]

            # if len(choices) == 3:
            #     c_3 += 1
            # elif len(choices) == 4:
            #     c_4 += 1
            # elif len(choices) == 5:
            #     c_5 += 1
            # else:
            #     error += 1

            # messo 4 perchè per tutti gli split sono le domande di numero maggiore 
            if len(choices) == 4:

                '''question'''
                question = instance["question"]["stem"]

                '''id (increasing number from 1 to 4) of the correct answer'''
                correct_answer_id = instance["answerKey"]

                '''save also question question subject'''
                subject = instance["info"]["subject"]

                language = instance["info"]["language"]

                # METTERE ANCHE LINGUA???

                '''create all the possible combinations (question, answer)'''

                if input_format == "0":
                    for c in choices:
                        content.append("{} {} {} {}".format(subject, language, question, c))
                elif input_format == "1":
                    for c in choices:
                        content.append("{} \\n {}".format(question, c))
                elif input_format == "2":
                    for c in choices:
                        content.append("{} {} {}".format(question, sep_token, c))
                
                answers = ["A", "B", "C", "D"]
                y = [0, 0, 0, 0]
                y[answers.index(correct_answer_id)] = 1
                labels += y

        # print(f"3 risposte: {c_3}")
        # print(f"4 risposte: {c_4}")
        # print(f"5 risposte: {c_5}")
        # print(f"errori: {error}")
            

                # a1 = choices[0]
                # a2 = choices[1]
                # a3 = choices[2]
                # a4 = choices[3]

                # if input_format == "0":
                #     content.append("Context: {} {} Question: {} {} Answer: {}".format(category, sep_token, question, sep_token, a1))
                #     content.append("Context: {} {} Question: {} {} Answer: {}".format(category, sep_token, question, sep_token, a2))
                #     content.append("Context: {} {} Question: {} {} Answer: {}".format(category, sep_token, question, sep_token, a3))
                #     content.append("Context: {} {} Question: {} {} Answer: {}".format(category, sep_token, question, sep_token, a4))
                # elif input_format == "1":
                #     content.append("{} \\n {} \\n {}".format(question, a1, category))
                #     content.append("{} \\n {} \\n {}".format(question, a2, category))
                #     content.append("{} \\n {} \\n {}".format(question, a3, category))
                #     content.append("{} \\n {} \\n {}".format(question, a4, category))
                # elif input_format == "2":
                #     content.append("<context>{}</context>\n<question>{}</question>\n<answer>{}</answer>".format(category, question, a1))
                #     content.append("<context>{}</context>\n<question>{}</question>\n<answer>{}</answer>".format(category, question, a2))
                #     content.append("<context>{}</context>\n<question>{}</question>\n<answer>{}</answer>".format(category, question, a3))
                #     content.append("<context>{}</context>\n<question>{}</question>\n<answer>{}</answer>".format(category, question, a4))
                # elif input_format == "3":
                #     content.append("Question: {} {} Answer: {} {} Context: {}".format(question, sep_token, a1, category, sep_token))
                #     content.append("Question: {} {} Answer: {} {} Context: {}".format(question, sep_token, a2, category, sep_token))
                #     content.append("Question: {} {} Answer: {} {} Context: {}".format(question, sep_token, a3, category, sep_token))
                #     content.append("Question: {} {} Answer: {} {} Context: {}".format(question, sep_token, a4, category, sep_token))

                # if correct_answer_id == "A":
                #     labels += [1, 0, 0, 0]
                # elif correct_answer_id == "B":
                #     labels += [0, 1, 0, 0]
                # elif correct_answer_id == "C":
                #     labels += [0, 0, 1, 0]
                # elif correct_answer_id == "D":
                #     labels += [0, 0, 0, 1]
                
        self.content, self.labels = content, labels
    
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index):
        s1, s2 = self.content[index], self.labels[index]
        return s1, s2
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
            
