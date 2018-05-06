#!/usr/bin/env python

import pickle
import numpy as np

import load_cifar10

print("Comming soon !")

data = open("./res/cifar-10-batches-py/data_batch_1", 'rb')
dict = pickle.load(data, encoding='bytes')

classes = load_cifar10.dictclass()

filenames = dict[b'filenames']
data = dict[b'data']
labels = np.asarray(dict[b'labels']).reshape((-1,1))
batch_label = dict[b'batch_label']

print(type(filenames))
print(data.shape)
print(labels.shape)
print(type(batch_label))

data = load_cifar10.normalize(data)
data = load_cifar10.toProperArray(data)
(data, labels) = load_cifar10.makeMiniBatch(data, labels, 150, False)

print(data[0].size())
print(labels[0].size())

fst = data[0][7]
print(classes[int(labels[0][7])])
for i in range(32):
    line = ""
    for j in range(32):
        line += " " + ("#" if (fst[0, i, j] + fst[1, i, j] + fst[2, i, j]) / 3.0 > 0.5 else ".")
    print(line)
