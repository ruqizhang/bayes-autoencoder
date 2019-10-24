import re
import os
import random
import tarfile
import codecs
from torchtext import data
SEED = 1
random.seed(8)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


class MR(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        """Create an MR dataset instance given a path and fields.
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        # text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]
        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with codecs.open(os.path.join(path, 'rt-polarity.neg'),'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with codecs.open(os.path.join(path, 'rt-polarity.pos'),'r','utf8') as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
        super(MR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, shuffle=True ,root='.',path="./datasets/MR/", **kwargs):
        """Create dataset objects for splits of the MR dataset.
        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        if shuffle: random.shuffle(examples)
        train_index = 4798
        test_index = 5331
        train_examples = examples[0:train_index] + examples[test_index:][0:train_index]
        test_examples = examples[train_index:test_index] + examples[test_index:][train_index:]

        random.shuffle(train_examples)
        random.shuffle(test_examples)
        print('train:',len(train_examples),'test:',len(test_examples))
        return (cls(text_field, label_field, examples=train_examples),
                cls(text_field, label_field, examples=test_examples)
                )

class MR_semi(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, numlabel,path=None, examples=None, **kwargs):

        fields = [('text', text_field), ('label', label_field)]
        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            file_neg = []
            file_pos = []
            count = 0
            cut = 4798-int(numlabel/2)
            with codecs.open(os.path.join(path, 'rt-polarity.neg'),'r','utf8') as f:
                for line in f:
                    file_neg.append(line)
            random.shuffle(file_neg)
            for line in file_neg:
                if count < cut:
                    examples += [
                        data.Example.fromlist([line, 'unlabelled'], fields)]
                    count += 1
                else:
                    examples += [
                        data.Example.fromlist([line, 'negative'], fields)]
            count = 0
            with codecs.open(os.path.join(path, 'rt-polarity.pos'),'r','utf8') as f:
                for line in f:
                    file_pos.append(line)
            random.shuffle(file_pos)
            for line in file_pos:
                if count < cut:
                    examples += [
                        data.Example.fromlist([line, 'unlabelled'], fields) ]
                    count += 1
                else:
                    examples += [
                        data.Example.fromlist([line, 'positive'], fields) ]
        # text_field = {'negative':0,'positive':1,'unlabelled':2}
        super(MR_semi, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, numlabel,shuffle=True ,root='.',path="./datasets/MR/", **kwargs):

        examples = cls(text_field, label_field,numlabel, path=path, **kwargs).examples
        unsup_index = 4798-int(numlabel/2)
        train_index = 4798
        test_index = 5331
        unsup_examples = examples[0:unsup_index] + examples[test_index:][0:unsup_index]
        train_examples = examples[unsup_index:train_index] + examples[test_index:][unsup_index:train_index]
        # dev_examples = examples[train_index:dev_index] + examples[test_index:][train_index:dev_index]
        test_examples = examples[train_index:test_index] + examples[test_index:][train_index:]
        random.shuffle(unsup_examples)
        random.shuffle(train_examples)
        # random.shuffle(dev_examples)
        random.shuffle(test_examples)
        print('unsup:',len(unsup_examples),'train:',len(train_examples),'test:',len(test_examples))
        return (cls(text_field, label_field, numlabel,examples=unsup_examples),
                cls(text_field, label_field, numlabel,examples=train_examples),
                cls(text_field, label_field, numlabel,examples=test_examples),
                )

class MR_semi_lstm(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):

        fields = [('text', text_field), ('label', label_field)]
        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            file_neg = []
            file_pos = []
            count = 0
            cut = 533+1500
            with codecs.open(os.path.join(path, 'rt-polarity.neg'),'r','utf8') as f:
                for line in f:
                    file_neg.append(line)
            random.shuffle(file_neg)
            for line in file_neg:
                if count < cut:
                    examples += [
                        data.Example.fromlist([line, 'negative'], fields)]
                count += 1
            count = 0
            with codecs.open(os.path.join(path, 'rt-polarity.pos'),'r','utf8') as f:
                for line in f:
                    file_pos.append(line)
            random.shuffle(file_pos)
            for line in file_pos:
                if count < cut:
                    examples += [
                        data.Example.fromlist([line, 'positive'], fields) ]
                count += 1

        super(MR_semi_lstm, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, shuffle=True ,root='.',path="./datasets/MR/", **kwargs):

        examples = cls(text_field, label_field, path=path, **kwargs).examples

        if shuffle: random.shuffle(examples)
        train_index = 1500
        test_index = 533+1500

        train_examples = examples[0:train_index] + examples[test_index:][0:train_index]
        # dev_examples = examples[train_index:dev_index] + examples[test_index:][train_index:dev_index]
        test_examples = examples[train_index:test_index] + examples[test_index:][train_index:]
        random.shuffle(train_examples)
        # random.shuffle(dev_examples)
        random.shuffle(test_examples)
        print('train:',len(train_examples),'test:',len(test_examples))
        return (cls(text_field, label_field, examples=train_examples),

                cls(text_field, label_field, examples=test_examples),
                )

# load MR dataset
def load_mr(text_field, label_field, batch_size):
    print('loading data')
    train_data,test_data = MR.splits(text_field, label_field)
    text_field.build_vocab(train_data, test_data)
    label_field.build_vocab(train_data, test_data)
    print('building batches')
    train_iter, test_iter = data.Iterator.splits(
        (train_data, test_data), batch_sizes=(batch_size, batch_size),repeat=False,
        device = -1
    )

    return train_iter, test_iter

def load_mr_semi(text_field, label_field, numlabel,batch_size):
    print('loading data')
    unsup_data, train_data,test_data = MR_semi.splits(text_field, label_field,numlabel)
    text_field.build_vocab(unsup_data,train_data, test_data)
    label_field.build_vocab(unsup_data,train_data, test_data)
    print('building batches')
    unsup_iter,train_iter, test_iter = data.Iterator.splits(
        (unsup_data,train_data, test_data), batch_sizes=(batch_size,batch_size, batch_size),repeat=False,
        device = -1
    )

    return unsup_iter,train_iter, test_iter

def load_mr_semi_lstm(text_field, label_field, batch_size):
    print('loading data')
    train_data,test_data = MR_semi_lstm.splits(text_field, label_field)
    text_field.build_vocab(train_data, test_data)
    label_field.build_vocab(train_data, test_data)
    print('building batches')
    train_iter, test_iter = data.Iterator.splits(
        (train_data, test_data), batch_sizes=(batch_size, batch_size),repeat=False,
        device = -1
    )

    return train_iter, test_iter
