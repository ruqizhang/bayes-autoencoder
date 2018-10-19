import numpy as np
import torch
import torchvision
import os

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

r"""This also includes marc's method for shuffling iterates with the unlab/lab loader categories."""

def getUnlabLoader(trainset, ul_BS, **kwargs):
    """ Returns a dataloader for the full dataset, with cyclic reshuffling """
    indices = np.arange(len(trainset))
    unlabSampler = ShuffleCycleSubsetSampler(indices)
    unlabLoader = DataLoader(trainset,sampler=unlabSampler,batch_size=ul_BS,**kwargs)
    return unlabLoader

def getLabLoader(trainset, lab_BS, amntLabeled=1, amntDev=0, **kwargs):
    """ returns a dataloader of class balanced subset of the full dataset,
        and a (possibly empty) dataloader reserved for devset
        amntLabeled and amntDev can be a fraction or an integer.
        If fraction amntLabeled specifies fraction of entire dataset to
        use as labeled, whereas fraction amntDev is fraction of labeled
        dataset to reserve as a devset  """
    numLabeled = amntLabeled
    if amntLabeled <= 1: 
        numLabeled *= len(trainset)
    numDev = amntDev
    if amntDev <= 1:
        numDev *= numLabeled

    labIndices, devIndices = classBalancedSampleIndices(trainset, numLabeled, numDev)

    labSampler = ShuffleCycleSubsetSampler(labIndices)
    labLoader = DataLoader(trainset,sampler=labSampler,batch_size=lab_BS,**kwargs)
    if numLabeled == 0: labLoader = EmptyLoader()

    devSampler = SequentialSubsetSampler(devIndices) # No shuffling on dev
    devLoader = DataLoader(trainset,sampler=devSampler,batch_size=50)
    return labLoader, devLoader

def classBalancedSampleIndices(trainset, numLabeled, numDev):
    """ Generates a subset of indices of y (of size numLabeled) so that
        each class is equally represented """
    y = np.array([target for img,target in trainset])
    uniqueVals = np.unique(y)
    numDev = (numDev // len(uniqueVals))*len(uniqueVals)
    numLabeled = ((numLabeled-numDev)// len(uniqueVals))*len(uniqueVals)
    
    classIndices = [np.where(y==val) for val in uniqueVals]
    devIndices = np.empty(numDev, dtype=np.int64)
    labIndices = np.empty(numLabeled, dtype=np.int64)
    
    dev_m = numDev // len(uniqueVals)
    lab_m = numLabeled // len(uniqueVals); assert lab_m>0, "Note: dev is subtracted from train"
    total_m = lab_m + dev_m
    for i in range(len(uniqueVals)):
        sampledclassIndices = np.random.choice(classIndices[i][0],total_m,replace=False)
        labIndices[i*lab_m:i*lab_m+lab_m] = sampledclassIndices[:lab_m]
        devIndices[i*dev_m:i*dev_m+dev_m] = sampledclassIndices[lab_m:]
        
    print("Creating Train, Dev split \
        with {} Train and {} Dev".format(numLabeled, numDev))
    return labIndices, devIndices

#TODO: change iter to single pass, add multi_iter method
class ShuffleCycleSubsetSampler(Sampler):
    """A cycle version of SubsetRandomSampler with
        reordering on restart """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return self._gen()

    def _gen(self):
        i = len(self.indices)
        while True:
            if i >= len(self.indices):
                perm = np.random.permutation(self.indices)
                i=0
            yield perm[i]
            i+=1
    
    def __len__(self):
        return len(self.indices)

class SequentialSubsetSampler(Sampler):
    """Samples sequentially from specified indices, does not cycle """
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

class EmptyLoader(object):
    """A dataloader that loads None tuples, with zero length for convenience"""
    def __next__(self):
        return (None,None)
    def __len__(self):
        return 0
    def __iter__(self):
        return self

def getUandLloaders(trainset, amntLabeled, lab_BS, ulab_BS, **kwargs):
    labLoader, _ = getLabLoader(trainset, lab_BS, amntLabeled, amntDev=0, **kwargs)
    ulabLoader = getUnlabLoader(trainset, ulab_BS, **kwargs)
    return {'lab': labLoader, 'ulab':ulabLoader}

def loaders(dataset, path, batch_size, num_workers, transform_train, transform_test, 
            unsup = False, use_validation=True, val_size=5000, shuffle_train=True,
            amntLabeled = 3000):

    ds = getattr(torchvision.datasets, dataset)
            
    path = os.path.join(path, dataset.lower())
    train_set = ds(root=path, train=True, download=True, transform=transform_train)

    if use_validation:
        print("Using train (" + str(len(train_set.train_data)-val_size) + ") + validation (" +str(val_size)+ ")")
        train_set.train_data = train_set.train_data[:-val_size]
        train_set.train_labels = train_set.train_labels[:-val_size]

        test_set = ds(root=path, train=True, download=True, transform=transform_test)
        test_set.train = False
        test_set.test_data = test_set.train_data[-val_size:]
        test_set.test_labels = test_set.train_labels[-val_size:]
        delattr(test_set, 'train_data')
        delattr(test_set, 'train_labels')
    else:
        print('You are going to run models on the test set. Are you sure?')
        test_set = ds(root=path, train=False, download=True, transform=transform_test)

    loaders_dict = {
                'train': torch.utils.data.DataLoader(
                    train_set,
                    batch_size=batch_size,
                    shuffle=True and shuffle_train,
                    num_workers=num_workers,
                    pin_memory=True
                ),
                'test': torch.utils.data.DataLoader(
                    test_set,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                ),
            }

    if not unsup:
        loaders_dict['train'] = getUandLloaders(loaders_dict['train'].dataset, amntLabeled, batch_size, batch_size)
        num_classes = max(train_set.train_labels) + 1
        return loaders_dict, num_classes.item()
    else:
        return loaders_dict