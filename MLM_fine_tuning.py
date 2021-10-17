import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import os

SEED = 51
LEARNING_RATES = [1e-4, 1e-5, 1e-6, 1e-7]

def prepare_data(test_mode=False,data_path='filtered_defs.csv'):
    """
    Reads data from csv, extracts the meanings and definitions from the Urban Dictionary entries
    and splits in a train and test set
    """
    defs = pd.read_csv(data_path)
    data = np.append(defs["meaning"].values, defs["example"].values)
    train, test = train_test_split(data, test_size=0.2, random_state=SEED)
    train = list(train)
    test = list(test)
    if test_mode:
        train = train[:100]
        test = test[:25]
    # remove nans
    train = [x for x in train if x == x]
    test = [x for x in test if x == x]
    return train, test

def preprocess(tokenizer, data, p):
    """
    Tokenize, add labels and randomly mask p% of the tokens
    """
    data_tokenized = tokenizer(data, padding=True, truncation=True, return_tensors="pt")
    # add labels key which is needed for torch Dataset
    data_tokenized['labels'] = data_tokenized.input_ids.detach().clone()

    rand = torch.rand(data_tokenized.input_ids.shape)
    # create mask array, do not mask special tokens
    mask_arr = (rand < p) * (data_tokenized.input_ids != 0) * \
               (data_tokenized.input_ids != 1) * (data_tokenized.input_ids != 2)
    data_tokenized = apply_masking(data_tokenized, mask_arr)
    return data_tokenized

def apply_masking(dataset, mask_arr):
    """
    Given a boolean masking array, apply the masking to the dataset inputs
    """
    selection = []
    for i in range(dataset.input_ids.shape[0]):
        selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())
    for i in range(dataset.input_ids.shape[0]):
        dataset.input_ids[i, selection[i]] = 50264
    return dataset

class Data(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, index):
        return {key: torch.tensor(val[index]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def epoch_time(start_time, end_time):
    """
    Translate start and end time to minutes and seconds
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_model(model, dataloader, optim):
    train_loss = 0
    model.train()
    for batch in dataloader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        loss = outputs[0]  # outputs.loss in documentation
        loss.backward()
        optim.step()
        train_loss += loss.item()
    return train_loss / len(dataloader)

def evaluate_model(model, dataloader):
    eval_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs[0]
            eval_loss += loss.item()
    return eval_loss / len(dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--data-path", type=str, default='data/UD_filtered_100000_sampled.csv')
    parser.add_argument("--small",type=bool, default=False)
    parser.add_argument("--maskp",type=float, default=0.15)
    parser.add_argument("--patience",type=int, default=3)
    parser.add_argument("--simplified-path",type=bool, default=True)

    args = parser.parse_args()

    if args.small:
        print("TRIAL WITH 100 SEQUENCES")

    # directories for saving models and results
    if not os.path.exists("models"):
        os.mkdir("models")
    if not os.path.exists("results/losses"):
        os.mkdir("results/losses")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print("------ PREPARING DATA ------")
    train, eval = prepare_data(test_mode=args.small,data_path=args.data_path)
    print("------ SUCCESSFULLY PREPARED DATA ------")
    print("------ PREPROCESS DATA ------")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', mask_token='<mask>')
    train_tokenized = preprocess(tokenizer, train, p=args.maskp)
    eval_tokenized = preprocess(tokenizer, eval, p=args.maskp)
    end_time = time.time()
    pre_mins, pre_secs = epoch_time(start_time, end_time)
    print("------ SUCCESSFULLY PREPROCESSED DATA AFTER {} MINS {} SECS ------".format(pre_mins, pre_secs))

    train_dataset = Data(train_tokenized)
    eval_dataset = Data(eval_tokenized)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size)
    evalloader = DataLoader(eval_dataset, batch_size=args.batch_size)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    MODEL_VAL_LOSSES = []
    for i, lr in enumerate(LEARNING_RATES):
        print("------ STARTING TRAINING MODEL {} ------".format(i+1))
        model = AutoModelForMaskedLM.from_pretrained("roberta-base")
        model.to(device)
        optim = AdamW(model.parameters(), lr=lr, betas = (0.9, 0.98), eps = 1e-6) #same betas as in RoBERTa paper
        run_time = 0
        min_eval_loss = 1e10
        epochs_no_improve = 0
        epoch_train_losses = []
        epoch_eval_losses = []
        scheduler = StepLR(optim, step_size=1, gamma=lr/args.num_epochs)
        for epoch in range(args.num_epochs):
            start_time = time.time()
            train_loss = train_model(model, trainloader, optim)
            epoch_train_losses.append(train_loss)
            eval_loss = evaluate_model(model, evalloader)
            epoch_eval_losses.append(eval_loss)
            if eval_loss < min_eval_loss:
                min_eval_loss = eval_loss
                epochs_no_improve = 1
            else:
                epochs_no_improve += 1
            scheduler.step()
            end_time = time.time()
            run_time += (end_time - start_time)
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print('Train loss for Epoch {}: {}'.format(epoch + 1, train_loss))
            print('Evaluation loss for Epoch {}: {}'.format(epoch + 1, eval_loss))
            print('Runtime for Epoch {}: {} mins {} secs'.format(epoch + 1, epoch_mins, epoch_secs))
            if epochs_no_improve >= args.patience:
                print("EARLY STOPPING")
                break
        print("RUNTIME FOR MODEL WITH LR {}: {:.2f} hours".format(lr, run_time/3600))
        MODEL_VAL_LOSSES.append(eval_loss)

        # save model and losses
        if args.simplified_path: model_save_path = "models/roberta_UD"
        else: model_save_path = "models/roberta_UD_lr"+str(lr)+"_epochs"+str(args.num_epochs)

        model.save_pretrained()
        textfile = open("losses/model_lr"+str(lr)+"_epochs"+str(args.num_epochs)+"_train.txt", "w")
        for elem in epoch_train_losses:
            textfile.write(str(elem) + "\n")
        textfile.close()
        textfile = open("losses/model"+str(lr)+"_epochs"+str(args.num_epochs)+"_eval.txt", "w")
        for elem in epoch_eval_losses:
            textfile.write(str(elem) + "\n")
        textfile.close()
        print(f"------ FINISHED TRAINING MODEL {i+1} ------")

    print("------ MODELS SUMMARY ------")
    for loss, lr in zip(MODEL_VAL_LOSSES, LEARNING_RATES):
        print("Evaluation loss is {:.5f} for learning rate {}".format(loss, lr))


