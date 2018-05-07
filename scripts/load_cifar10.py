import pickle
import numpy as np
import torch as th

def load_data_labels(nbbatch):
    all_data = []
    all_labels = []
    for i in range(nbbatch):
        data = open("./res/cifar-10-batches-py/data_batch_%s" % (i + 1), 'rb')
        dict = pickle.load(data, encoding='bytes')
        data = dict[b'data']
        labels = np.asarray(dict[b'labels']).reshape((-1,1))
        all_data.append(data)
        all_labels.append(labels)
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return (all_data, all_labels)

def dictclass():
    return {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}

def toProperArray(data):
    all_red = data[:,:1024]
    all_green = data[:,1024:2048]
    all_blue = data[:,2048:]
    return np.stack([all_red, all_green, all_blue], axis=1).reshape((-1, 3, 32, 32))

def normalize(data):
    return np.divide(data, 255)

def toFloatTensor(array, use_cuda):
    if use_cuda:
        return th.cuda.FloatTensor(array)
    else:
        return th.FloatTensor(array)

def toLongTensor(array, use_cuda):
    if use_cuda:
        return th.cuda.LongTensor(array)
    else:
        return th.LongTensor(array)

def makeMiniBatch(numpyArrayImg, numpyArrayLabels, batch_size, use_cuda):
    resImg = []
    resLabels = []
    cpt = 0
    length = numpyArrayImg.shape[0]
    while cpt < length:
        size = batch_size if length - cpt >= batch_size else length - cpt
        tmpImgs = numpyArrayImg[cpt:cpt+size]
        tmpImgs = toFloatTensor(tmpImgs, use_cuda)
        tmpLabels = numpyArrayLabels[cpt:cpt+size]
        tmpLabels = toLongTensor(tmpLabels, use_cuda)
        resImg.append(tmpImgs)
        resLabels.append(tmpLabels.view(-1))
        cpt += size
    return (resImg, resLabels)
