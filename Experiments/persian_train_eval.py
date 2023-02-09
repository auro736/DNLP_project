from Utils.utils import *
from Utils.custom_parser import my_parser
from Models.model import Model
from Ds.persian_ds import PersianDataset

import time 
from pathlib import Path
import optuna


from torch.utils.data import DataLoader

def build_model(lr):

    args = my_parser()
    print(args)

    eval_batch_size = args.eval_bs
    name = args.name
    num_choices = args.num_choices

    model = Model(
        name=name,
        num_choices=num_choices
    ).cuda()


    optimizer = configure_optimizer(model, args, lr)

    return model, optimizer

def train_eval(model, optimizer, train_batch_size, save = False):

    #train_batch_size = args.bs
    eval_batch_size = args.eval_bs
    epochs = args.epochs
    shuffle = args.shuffle
    input_format = args.input_format
    name = args.name

    sep_token = model.tokenizer.sep_token

    json_path_train = "/content/DNLP_project/data/persian/train.jsonl"
    json_path_valid = "/content/DNLP_project/data/persian/valid.jsonl"

    train_dataset = PersianDataset(
                        json_path_train, 
                        sep_token=sep_token, 
                        input_format=input_format, 
                        shuffle=True
                        )
    
    val_dataset = PersianDataset(
                        json_path_valid, 
                        sep_token=sep_token, 
                        input_format=input_format, 
                        shuffle=False
                        )
    
    train_loader = DataLoader(
                        train_dataset, 
                        shuffle=shuffle, 
                        batch_size=train_batch_size,
                        collate_fn=train_dataset.collate_fn
                        )

    val_loader = DataLoader(
                        val_dataset, 
                        shuffle=False, 
                        batch_size=eval_batch_size, 
                        collate_fn=val_dataset.collate_fn
                        )


    # if "/" in name:
    #     sp = name[name.index("/") + 1:]
    # else:
    #     sp = name

    # exp_id = str(int(time.time()))
    # vars(args)["exp_id"] = exp_id
    # rs = "Acc: {}"

    # path = "/content/DNLP_project/log/persian/" + exp_id + "/" + name.replace("/", "-")
    # Path("/content/DNLP_project/log/persian/" + exp_id + "/").mkdir(parents=True, exist_ok=True)

    # fname = "/content/DNLP_project/log/persian/" + exp_id + "/" + "args.txt"

    # f = open(fname, "a")
    # f.write(str(args) + "\n\n")
    # f.close()

    # Path("/content/DNLP_project/log/persian/").mkdir(parents=True, exist_ok=True)
    # lf_name = "/content/DNLP_project/log/persian/" + name.replace("/", "-") + ".txt"
    # lf = open(lf_name, "a")
    # lf.write(str(args) + "\n\n")
    # lf.close()

    # path = "saved/persian_dataset/" + exp_id + "/" + name.replace("/", "-")
    # Path("saved/persian_dataset/" + exp_id + "/").mkdir(parents=True, exist_ok=True)

    # fname = "saved/persian_dataset/" + exp_id + "/" + "args.txt"

    # f = open(fname, "a")
    # f.write(str(args) + "\n\n")
    # f.close()

    # Path("results/persian_dataset/").mkdir(parents=True, exist_ok=True)
    # lf_name = "results/persian_dataset/" + name.replace("/", "-") + ".txt"
    # lf = open(lf_name, "a")
    # lf.write(str(args) + "\n\n")
    # lf.close()

    #val_ins_acc_list = list()

    start_time = time.time()
    for e in range(epochs):
        
        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer)
        val_loss, val_acc, val_ins_acc, val_f1 = eval(model, val_loader)

        x = "Epoch {}: Loss: Train {}; Val {}".format(e + 1, train_loss, val_loss)
        y1 = "Classification Acc: Train {}; Val {}".format(train_acc, val_acc)
        y2 = "Classification Macro F1: Train {}; Val {}".format(train_f1, val_f1)
        z = "Instance Acc: Val {}".format(val_ins_acc)

        print(x)
        print(y1)
        print(y2)
        print(z)

        # lf = open(lf_name, "a")
        # lf.write(x + "\n" + y1 + "\n" + y2 + "\n" + z + "\n\n")
        # lf.close()

        # f = open(fname, "a")
        # f.write(x + "\n" + y1 + "\n" + y2 + "\n" + z + "\n\n")
        # f.close()
    
    # training_time = time.time() - start_time
    # print('Training time:', training_time )
    # lf = open(lf_name, "a")
    # lf.write('Training time: {}'.format(training_time) + "\n")
    # lf.close()
    

    # print("Testing...")

    # print("Results for test LIT")
    # start_time = time.time()
    # test_preds_lit, ins_acc_lit = train_or_eval_model(model, test_loader_lit, split="Test")
    # print('Execution time:', time.time() - start_time)

    # print("Results for test CK")
    # start_time = time.time()
    # test_preds_ck, ins_acc_ck = train_or_eval_model(model, test_loader_ck, split="Test")
    # print('Execution time:', time.time() - start_time)

    # print("Results for test ML")
    # start_time = time.time()
    # test_preds_ml, ins_acc_ml  = train_or_eval_model(model, test_loader_ml, split = "Test")
    # print('Execution time:', time.time() - start_time)

    # with open(path + "-epoch-" + str(e + 1) + ".txt", "w") as f:
    #         f.write("\n".join(list(test_preds)))

    # lf = open(lf_name, "a")
    # lf.write("Instance Acc: Test LIT {}".format(ins_acc_lit)+"\n")
    # lf.write("Instance Acc: Test CK {}".format(ins_acc_ck)+"\n")
    # lf.write("Instance Acc: Test ML {}".format(ins_acc_ml)+"\n")
    # lf.write("-" * 100 + "\n")
    # lf.close()

    return val_ins_acc


def objective(trial):

    params = {
        'learning_rate' : trial.suggest_categorical('learning_rate', [3e-5, 5e-5]),
        'batch_size' : trial.suggest_categorical('batch_size', [8,16])
    }

    model, optimizer = build_model(lr = params["learning_rate"])

    accuracy = train_eval(model, optimizer, train_batch_size = params["batch_size"],save = False)

    return accuracy

if __name__ == "__main__":
    
    study = optuna.create_study(direction = "maximize", sampler = optuna.samplers.GridSampler())
    study.optimize(objective, n_trials = 4)

    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))

    model, optimizer = build_model(lr = best_trial.params["learning_rate"])
    accuracy = train_eval(model, optimizer, train_batch_size = best_trial.params["batch_size"],save = True)
    
    
