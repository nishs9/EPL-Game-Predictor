import os.path
import pandas as pd
pd.set_option('display.max_columns',None)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

currDirectory = os.path.abspath(os.path.dirname(__file__))

raw_data = pd.read_csv(os.path.join(currDirectory,"../data/E0-2018-2019.csv"))
raw_data2 = pd.read_csv(os.path.join(currDirectory,"../data/E0-2019-2020.csv"))
raw_data = pd.concat([raw_data,raw_data2],axis=0)