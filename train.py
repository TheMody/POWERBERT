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
SEED = 999
random.seed(SEED)
np.random.seed(SEED)

ACTIVE_METRIC_NAME = 'accuracy'
REWARD_ATTR_NAME = 'objective'
datasets = [ "mnli","cola", "sst2", "mrpc","qqp", "rte"]#"qqp", "rte" 
eval_ds = [ "rtesmall", "qqpsmall","qqp", "rte"]




def train(args, config):

    torch.multiprocessing.set_start_method('spawn', force=True)
    max_epochs = int(config["DEFAULT"]["epochs"])

    batch_size = int(config["DEFAULT"]["batch_size"])

    dataset = config["DEFAULT"]["dataset"]
    print("dataset:", dataset)
    log_file = config["DEFAULT"]["directory"]+"/log_file.csv"
    load = config["DEFAULT"]["load"] == "True"
    
        
    print("running baseline")
    num_classes = 2
    if "mnli" in dataset:
        num_classes = 3
    num_classes2 = 2
    #  print("loading model")

  
        
    device = torch.device('cuda')
    model = TransformerPredictor().to(device)
    model.fit()


        

    torch.cuda.empty_cache()

