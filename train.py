import torch

import numpy as np
import random
from embedder import TransformerPredictor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from data import load_data, load_wiki
# Fixing seed for reproducibility
from transformers import BertTokenizer
SEED = 999
random.seed(SEED)
np.random.seed(SEED)

datasets = [ "mnli","cola", "sst2", "mrpc","qqp", "rte"]#"qqp", "rte" 
eval_ds = [ "rtesmall", "qqpsmall","qqp", "rte"]




def train(args, config):

    torch.multiprocessing.set_start_method('spawn', force=True)
    max_epochs = int(config["DEFAULT"]["epochs"])

    batch_size = int(config["DEFAULT"]["batch_size"])

    dataset = config["DEFAULT"]["dataset"]
    print("dataset:", dataset)
    log_file = config["DEFAULT"]["directory"]+"/log_file.csv"
    reduce = config["DEFAULT"]["reduce"] == "True"
    
    print("loading dataset")
    ds = load_wiki()
    
    print("building the transformer")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda')
    model = TransformerPredictor(tokenizer.vocab_size, 768, tokenizer.vocab_size,12,12, batch_size= batch_size, reduce=reduce).to(device)
    print("fitting")
    model.fitmlm(ds, max_epochs)
    torch.save(model, config["DEFAULT"]["directory"]+"/model.pt")


        

    torch.cuda.empty_cache()

def test(args, config):

    dataset = config["DEFAULT"]["dataset"]
    print("dataset:", dataset)
    X,X_val, _, y, y_val, _ = load_data(dataset)
    max_epochs = int(config["DEFAULT"]["epochs"])
    num_classes = 2
    model = torch.load(config["DEFAULT"]["load"]+"/model.pt")
    model.lr = 2e-5
    model.fit(X,y, max_epochs ,num_classes )
    acc = model.evaluate(X_val, y_val)
    print("acc", acc)

