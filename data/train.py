import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model.model import NeuralNet


with open('./datasets/intents.json', 'r') as f:
    intents = json.load(f)
    
all_words = []
tags = [] 
xy = [] # THIS LIST WILL CONTAIN THE PATTERNS AND TAGS

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w) 
        xy.append((w, tag))
        
ignore_words = ['?', '!', '.', ',', ':', ';', '-'] # THIS LIST WILL CONTAIN THE WORDS THAT WILL BE IGNORED
all_words = [stem(w) for w in all_words if w not in ignore_words] # STEMMING
all_words = sorted(set(all_words)) # SET WILL REMOVE DUPLICATES
tags = sorted(set(tags)) # SET WILL REMOVE DUPLICATES

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
 
# HYPERPARAMETERS
batch_size = 8
num_workers = 2
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

model = NeuralNet(input_size, hidden_size, output_size)
    