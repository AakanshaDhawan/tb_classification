import numpy as np
import pickle
import random


def getdata():
    # read data and shuffle
    index=[i for i in range(662)]

    f=open("./data_train","rb+")
    data=pickle.load(f)
    data=data[index]
    data_train=data[0:]
    f=open("./train_y","rb+")
    box=pickle.load(f)
    box=box[index]
    train_y=box[0:]

    index=[i for i in range(124)]
    f=open("./data_test","rb+")
    data=pickle.load(f)
    data=data[index]
    data_test=data[0:]
    f=open("./train_y","rb+")
    box=pickle.load(f)
    box=box[index]
    test_y=box[0:]
    
    return data_train,train_y,data_test,test_y
