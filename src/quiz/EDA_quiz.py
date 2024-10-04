import numpy as np
import pandas as pd


candy_data = pd.read_csv('../../data/quiz_data_sources/candyhierarchy2017.csv',encoding="ISO-8859-1")
print("***********\nHead\n***********",candy_data.head())
print("***********\nDescribe\n***********",candy_data.describe())
print("***********\nColumns\n***********",candy_data.columns)
print("***********\nShape\n***********",candy_data.shape)
print("***********\nNulls\n***********",candy_data.isnull().sum())
print("***********\nAGE Nulls\n***********",candy_data['Q3: AGE'].isnull().sum())