import nltk
from nltk.stem.lancaster import LancasterStemmer
from torch import classes, le
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow
import random
import json
import os.path
import os
import pickle
from datetime import datetime
from datetime import date

def write_vars():
    words = []
    labels = []
    docs = []
    docs_y = []
    docs_x = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs.append(pattern)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
            
            if intent["tag"] not in labels:
                labels.append(intent["tag"])
                
    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    # words = [stemmer.stem(w.lower()) for w in words if w not in "?" if w not in "!"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x,doc in enumerate(docs_x):
        bag = []
        
        wrds = [stemmer.stem(w) for w in doc]
        
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
                
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)    

with open("psycho.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    write_vars()
    

# tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

def reload_vars():
    os.remove("data.pickle")
    write_vars()

def train_model():
    # log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # tensorboard_callback2 = tensorflow.keras.callbacks.TensorBoard(log_dir='./log_dir', histogram_freq=0, write_graph=True, write_images=True)
    # tbCallBack = tensorflow.keras.callbacks.TensorBoard(log_dir='./logs/', histogram_freq=0, write_graph=True, write_images=True)
    reload_vars()
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] = 1
                
    return np.array(bag)

def chat():
    RUN = True
    print("Start talking with the bot! (type quit to stop)")
    while RUN == True:
        inp = input("You: ")
        if inp.lower() == "quit":
            RUN = False
        if inp.lower() == "train":
            train_model()
            
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]
        # print(tag)
        
        if results[results_index] > 0.7:
            for tag2 in data["intents"]:
                if tag2["tag"] == tag:
                    responses = tag2["responses"]
         
            if tag == "goodbye":
                print("AI: ",random.choice(responses))
                RUN = False
            
            if tag == "time":
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(f"AI: {random.choice(responses)} {current_time}")
                
            if tag == "date":
                today = date.today()
                print(f"AI: {random.choice(responses)} {today}")
            
            else:
                print("AI: ", random.choice(responses))    
            # print(random.choice(responses))
            
        else:
            print("AI: I didnt get that, try again.")
                
            

chat()