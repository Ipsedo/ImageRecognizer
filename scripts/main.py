#!/usr/bin/env python

import pickle
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.autograd as ag

import load_cifar10
import model

use_cuda = th.cuda.is_available()

# Load train data
# data = open("./res/cifar-10-batches-py/data_batch_1", 'rb')
# dict = pickle.load(data, encoding='bytes')
#
# classes = load_cifar10.dictclass()
#
# filenames = dict[b'filenames']
# data = dict[b'data']
# labels = np.asarray(dict[b'labels']).reshape((-1,1))
# batch_label = dict[b'batch_label']
#
# print(type(filenames))
# print(data.shape)
# print(labels.shape)
# print(type(batch_label))

print("Load data...")
(data, labels) = load_cifar10.load_data_labels(5)

batch_size = 150

data = load_cifar10.normalize(data)
data = load_cifar10.toProperArray(data)
(data, labels) = load_cifar10.makeMiniBatch(data, labels, batch_size, use_cuda)

# print(data[0].size())
# print(labels[0].size())
#
# fst = data[0][7]
# print(classes[int(labels[0][7])])
# for i in range(32):
#     line = ""
#     for j in range(32):
#         line += " " + ("#" if (fst[0, i, j] + fst[1, i, j] + fst[2, i, j]) / 3.0 > 0.5 else ".")
#     print(line)

# Load test data
data_test = open("./res/cifar-10-batches-py/test_batch", 'rb')
dict_test = pickle.load(data_test, encoding='bytes')

data_test = dict_test[b'data']
labels_test = np.asarray(dict_test[b'labels']).reshape((-1,1))
data_test = load_cifar10.normalize(data_test)
data_test = load_cifar10.toProperArray(data_test)

def eval_model(model, data, labels):
    model.eval()
    err = 0
    total = 0
    for img, y in zip(data, labels):
        x = load_cifar10.toFloatTensor(img, use_cuda).unsqueeze(0)
        x = ag.Variable(x)
        out = model(x)
        _, out = th.max(out, 1)
        err += 1 if out.item() != y else 0
        total += 1
    print("Test on %s img, (err / total) : %s / %s" % (len(data), err, total))

print("Build model...")
EPOCH = 30
learning_rate = 2e-3
model = model.ConvModel(32, 10)
loss_fn = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if use_cuda:
    model.cuda()
    loss_fn.cuda()

print("Train model...")
for i in range(EPOCH):
    model.train()
    total_loss = 0
    for x, y in zip(data, labels):
        model.zero_grad()
        x = ag.Variable(x)
        y = ag.Variable(y)
        out = model(x)
        loss = loss_fn(out, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print("Epoch %s : loss %s" % (i, total_loss))
    eval_model(model, data_test, labels_test)
