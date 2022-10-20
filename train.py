import torch

import numpy as np
import random
from embedder import TransformerPredictor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from data import load_data, load_wiki, load_wikiandbook
# Fixing seed for reproducibility
from transformers import BertTokenizer

SEED = 999
random.seed(SEED)
np.random.seed(SEED)

datasets = [ "mnli","cola", "sst2", "mrpc","qqp", "rte"]#"qqp", "rte" 
eval_ds = [ "rtesmall", "qqpsmall","qqp", "rte"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args, config):

    torch.multiprocessing.set_start_method('spawn', force=True)
    max_epochs = int(config["DEFAULT"]["epochs"])
    max_steps = int(config["DEFAULT"]["steps"])

    batch_size = int(config["DEFAULT"]["batch_size"])

    dataset = config["DEFAULT"]["dataset"]
    print("dataset:", dataset)
    log_file = config["DEFAULT"]["directory"]+"/log_file.csv"
    reduce = config["DEFAULT"]["reduce"] == "True"
    
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("loading dataset")
    ds = load_wikiandbook(batch_size, tokenizer, 256)
    
    print("building the transformer")

    model = TransformerPredictor(tokenizer.vocab_size, 768, tokenizer.vocab_size,12,12, batch_size= batch_size, reduce=reduce).to(device)
    print("fitting")
    model.fitmlm(ds, max_steps, config["DEFAULT"]["directory"]+"/model.pt")
  #  torch.save(model, config["DEFAULT"]["directory"]+"/model.pt")
  #  model = torch.load(config["DEFAULT"]["directory"]+"/model.pt")
   # model.fitmlm(ds, max_steps)#, config["DEFAULT"]["directory"]+"/model.pt")

        

    torch.cuda.empty_cache()

def test(args, config):
    reduce = config["DEFAULT"]["reduce"] == "True"
    batch_size = int(config["DEFAULT"]["batch_size"])
    dataset = config["DEFAULT"]["dataset"]
    print("dataset:", dataset)
    X,X_val, _, y, y_val, _ = load_data(dataset)
    max_epochs = int(config["DEFAULT"]["epochs"])
    num_classes = 2
    # model = torch.load(config["DEFAULT"]["load"]+"/model.pt")
    # model.lr = 2e-5
    # model.batch_size = batch_size
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TransformerPredictor(tokenizer.vocab_size, 768, num_classes,12,12, batch_size= batch_size, reduce=reduce, lr = 2e-5).to(device)
   # ds = load_wiki()
   # model.fitmlm(ds, max_epochs, config["DEFAULT"]["directory"]+"/model.pt")
    model.fit(X,y, max_epochs ,num_classes )
  #  model.fit(X,y, max_epochs ,num_classes )
    torch.save(model, config["DEFAULT"]["directory"]+"/modelfine.pt")
    acc = model.evaluate(X_val, y_val)
    print("acc", acc)

