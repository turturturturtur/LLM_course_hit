import torch
import os.path as osp
import numpy as np
import pandas as pd
from hf import AutoTokenizer

# Read from kaggle data, which can be loaded by python( pandans and numpy can't)
file_path = "Medical_Text"

with open(osp.join(file_path,'train.dat')) as file:
    train_dataset = [f.strip() for f in file]

with open(osp.join(file_path,'test.dat')) as file:
    test_dataset = [f.strip() for f in file]

# load tokenizor