import json
import random
import pandas as pd
from tqdm import tqdm 
from transformers import pipeline

def eng_to_fr(file_path, split):

    print("Split: ",split)

    x = open(file_path).readlines()
    en_fr_translator = pipeline("translation_en_to_fr")
    
    for line in tqdm(x):

        instance = json.loads(line)

        dict_ = {}
        dict_["question"] = {}
        
        dict_["question"]["stem"] = en_fr_translator(instance["question"]["stem"])[0]["translation_text"]
        dict_["question"]["choices"] = list()

        for a in instance["question"]["choices"]:
            en_text = a["text"]
            fr_text = en_fr_translator(en_text)[0]["translation_text"]
            en_label = a["label"]
            tmp = {"text":fr_text, "label": en_label}
            dict_["question"]["choices"].append(tmp)
        
        if "answerKey" in instance:
            dict_["answerKey"] = instance["answerKey"]
        else:
            dict_["answerKey"] = "A"

        saving_fp = f"/mnt/c/Users/auror/Desktop/NUOVA REPO PROGETTO DNLP/DNLP_project/data/qasc/FR/{split.lower()}_FR.jsonl"
        
        #saving_fp = f"/content/DNLP_project/data/qasc/FR/{split.lower}_FR.jsonl"

        with open(saving_fp, "a", encoding='utf8') as outfile:
            #outfile.write(json_object)
            json.dump(dict_,outfile,ensure_ascii=False)
            outfile.write('\n')



if __name__ == "__main__":

    json_path_train = "/mnt/c/Users/auror/Desktop/NUOVA REPO PROGETTO DNLP/DNLP_project/data/qasc/train.jsonl"
    json_path_valid = "/mnt/c/Users/auror/Desktop/NUOVA REPO PROGETTO DNLP/DNLP_project/data/qasc/dev.jsonl"
    json_path_test = "/mnt/c/Users/auror/Desktop/NUOVA REPO PROGETTO DNLP/DNLP_project/data/qasc/test.jsonl"

    # json_path_train = "/content/DNLP_project/data/qasc/train.jsonl"
    # json_path_valid = "/content/DNLP_project/data/qasc/dev.jsonl"
    # json_path_test = "/content/DNLP_project/data/qasc/test.jsonl"

    #eng_to_fr(json_path_train,split = "Train")
    #eng_to_fr(json_path_valid,split = "Dev")
    eng_to_fr(json_path_test,split = "Test")


       