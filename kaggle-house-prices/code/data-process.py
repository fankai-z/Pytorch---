# %matplotlib inline
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys

sys.path.append("..")

print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')
print(train_data.shape)
test_data.shape

a = [1, 2, 3, 'NaN']
b = ["a", "b", "c", "NaN"]
df = pd.DataFrame({"A": a, "B": b})
mm = df.dtypes.index
print(mm)
print("===")
numerical_index = df.dtypes[df.dtypes != 'object'].index
print(df.dtypes)
print(numerical_index)


def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net






