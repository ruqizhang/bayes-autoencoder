import os
import torch
from itertools import starmap

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

def get_batch(i, dataset, bptt):
    seq_len = min(bptt, len(dataset) - 1 - i)

    data = dataset[i:i+seq_len]
    target = dataset[i+1:i+1+seq_len].view(-1)

    return data, target

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

def loaders(dataset, path, batch_size, bptt, use_cuda = True):
    if dataset != 'ptb':
        print('Only ptb is implemented currently')
        

    corpus = Corpus(path)

    train_data = batchify(corpus.train, batch_size, use_cuda)
    val_data = batchify(corpus.valid, batch_size, use_cuda)
    test_data = batchify(corpus.test, batch_size, use_cuda)

    loaders = {'train': TextDataLoader(train_data, bptt),
                'valid': TextDataLoader(val_data, bptt),
                'test': TextDataLoader(test_data, bptt)}

    ntokens = len(corpus.dictionary)

    return loaders, ntokens
