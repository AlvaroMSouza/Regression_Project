import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Regression Problem
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanSquaredError

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



df = pd.read_csv('london_houses.csv')

print(df.head(5))