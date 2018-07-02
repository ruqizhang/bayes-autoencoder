import numpy as np
import torch
import torchvision
import os

def loaders(dataset, path, batch_size, num_workers, transform_train, transform_test, unsup = False, use_validation=True, val_size=5000, shuffle_train=True):

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
        num_classes = max(train_set.train_labels) + 1
        return loaders_dict, num_classes
    else:
        return loaders_dict