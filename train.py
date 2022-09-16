import torch
import torch.nn as nn

# AutoGluon and HPO tools
import autogluon.core as ag
import pandas as pd
import numpy as np
import random
import math
from embedder import TransformerPredictor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
import time
from data import load_data, SimpleDataset, load_wiki
from torch_geometric.loader import DataLoader
from plot import plot_TSNE_clustering
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder
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

