import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from model.model import NeuralNet
# IMPORT CALLBACK FROM BS3
from stable_baselines3.common.callbacks import BaseCallback



if __name__ == '__main__':
    # freeze_support()
    
    
    
    # class TrainAndLoggingCallback(BaseCallback):
        
    #     def __init__(self, check_freq, save_path, verbose=1):
    #         super(TrainAndLoggingCallback, self).__init__(verbose)
    #         self.check_freq = check_freq
    #         self.save_path = save_path

    #     def _init_callback(self):
    #         if self.save_path is not None:
    #             os.makedirs(self.save_path, exist_ok=True)

    #         return True
        
    # CHECKPOINT_DIR = './train/train_basic'
    # LOG_DIR = './logs/'
    
     
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
    num_workers = 0
    hidden_size = 8
    output_size = len(tags)
    input_size = len(X_train[0])
    learning_rate = 0.001
    num_epochs = 10000

    # OTHER STUFF
    epoch_display_rate = 100

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # LOSS AND OPTIMIZIER
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # TRAINING LOOP
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            
            # FORWARD THE DATA
            output = model(words)
            
            
            loss = criterion(output, labels)
            
            # BACKWARD AND OPTIMIZER STEP
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch +1) % epoch_display_rate == 0:
            print(f'epoch={epoch+1}/{num_epochs}, loss={loss.item():.10f} ')

    print(f'final loss={loss.item():.10f} ')
            