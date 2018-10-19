import os
import torch
from itertools import starmap
import torchvision

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'ptb.test.txt'))
        #IMDB
        # self.train = self.tokenize(os.path.join(path, 'imdb_train40.txt'))
        # self.test = self.tokenize(os.path.join(path, 'imdb_test40.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

def batchify(data, bsz, use_cuda = True):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if use_cuda:
        data = data.cuda()
    return data

class TextDataLoader(object):
    def __init__(self, dataset, bptt):
        self.bptt = bptt
        self.dataset = dataset
        
    def get_batch(self, i):
        seq_len = min(self.bptt, len(self.dataset) - 1 - i)

        data = self.dataset[i:i+seq_len]
        target = self.dataset[i+1:i+1+seq_len].view(-1)

        return data, target

    def __iter__(self):
        return starmap(self.get_batch, zip(range(0, self.dataset.size(0) - 1, self.bptt)))

    def __len__(self):
        return len(self.dataset) * self.bptt

def construct_text_loaders(dataset, path, batch_size, bptt, transform_train, transform_test,
                            use_validation=True, use_cuda = True):
    corpus = Corpus(path)

    train_data = batchify(corpus.train, batch_size, use_cuda)
    val_data = batchify(corpus.valid, batch_size, use_cuda)
    test_data = batchify(corpus.test, batch_size, use_cuda)

    if not use_validation:
        print('Warning, ptb has default validation set so it will still be returned.')

    loaders = {'train': TextDataLoader(train_data, bptt),
                'valid': TextDataLoader(val_data, bptt),
                'test': TextDataLoader(test_data, bptt)}

    ntokens = len(corpus.dictionary)

    return loaders, ntokens

class ImageDataLoader(torch.utils.data.DataLoader):
    def __init__(self,**kwargs):
        super(ImageDataLoader,self).__init__(**kwargs)
    def __len__(self):
        return len(self.dataset)

def construct_image_loaders(dataset, path, batch_size, bptt, transform_train, transform_test, 
                            use_validation=True, use_cuda = True, num_workers=4, val_size=5000):
    ds = getattr(torchvision.datasets, dataset)
            
    path = os.path.join(path, dataset.lower())
    train_set = ds(root=path, train=True, download=True, transform=transform_train)

    if use_validation:
        print("Using train (" + str(len(train_set.train_data)-val_size) + ") + validation (" +str(val_size)+ ")")
        train_set.train_data = train_set.train_data[:-val_size]
        
        test_set = ds(root=path, train=True, download=True, transform=transform_test)
        test_set.train = False
        test_set.test_data = test_set.train_data[-val_size:]
        test_set.test_labels = test_set.train_labels[-val_size:]
        delattr(test_set, 'train_data')
        delattr(test_set, 'train_labels')
    else:
        print('You are going to run models on the test set. Are you sure?')
        test_set = ds(root=path, train=False, download=True, transform=transform_test)

    train_loader = ImageDataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = ImageDataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                )

    #def test_loader.__len__(self):
    #    return len(self.dataset)

    loaders_dict = {'train':train_loader, 'test': test_loader}
    
    return loaders_dict, len(train_set.train_data)

    
def loaders(dataset, path, batch_size, bptt, transform_train, transform_test, use_validation, use_cuda = True):
    if dataset == 'ptb':        
        return construct_text_loaders(dataset, path, batch_size, bptt, transform_train, transform_test, use_validation, use_cuda = True)
    if dataset == 'MNIST':
        return construct_image_loaders(dataset, path, batch_size, bptt, transform_train, transform_test, use_validation, use_cuda = True)
