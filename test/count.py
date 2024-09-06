# # from torch import nn 
# # from torch.optim import Adam 
# # from torch.utils.data import dataloader
# # from torchvision import datasets
# # from torchvision.transforms import ToTensor

# # train
# import matplotlib.pyplot as plt
# import numpy as np

# # Raw data string
# data_line = "18,0.358139535,0.509090909,0.320930233,0.472727273,0.297674419,0.454545455,0.274418605,0.463636364,0.246511628,0.536363636,0.227906977,0.672727273,0.227906977,0.831818182,0.269767442,0.963636364,0.311627907,1,0.358139535,0.963636364,0.395348837,0.9,0.423255814,0.859090909,0.451162791,0.854545455,0.488372093,0.886363636,0.544186047,0.918181818,0.618604651,0.954545455,0.711627907,0.977272727,0.795348837,0.972727273,0.869767442,0.945454545,0.925581395,0.886363636,0.953488372,0.809090909,0.944186047,0.718181818,0.911627907,0.627272727,0.855813953,0.563636364,0.795348837,0.540909091,0.730232558,0.554545455,0.674418605,0.6,0.651162791,0.654545455,0.665116279,0.709090909,0.71627907,0.754545455,0.795348837,0.772727273,0.879069767,0.754545455,0.948837209,0.722727273,0.176744186,0.4,0.102325581,0.377272727,0.041860465,0.327272727,0,0.272727273,0.009302326,0.213636364,0.069767442,0.163636364,0.153488372,0.145454545,0.302325581,0.163636364,0.413953488,0.195454545,0.548837209,0.181818182,0.637209302,0.109090909,0.706976744,0.022727273,0.813953488,0,0.897674419,0.013636364,1,0.045454545"

# # Convert the string data into a list of floats
# data_values = [float(x) for x in data_line.split(',') if x]

# # The first value is the label, and the rest are feature values
# label = int(data_values[0])
# features = data_values[1:]

# # Convert features into a numpy array for plotting
# features_array = np.array(features)

# # Create a plot
# plt.figure(figsize=(12, 6))
# plt.plot(features_array, marker='o', linestyle='-', color='b')
# plt.title(f'Plot of Features for Label {label}')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.grid(True)
# plt.show()



import pandas as pd 

file_path = r'D:\I4-internship\Internship\data\train_data.csv' 

df = pd.read_csv(file_path,header=None)



label_counts = df.iloc[:,0].value_counts()

Alllabel_counts = df.sum(axis=0)
print(Alllabel_counts)
print(len(df))
