import sys
sys.path.append('..')
import unsup_text.data as text_data
corpus = text_data.Corpus('/scratch/datasets/ptb')

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    #if args.cuda:
    data = data.cuda(async=True)
    return data

loader_batches = batchify(corpus.test, 64)[0:35, :]

def get_batch(i, source=loader_batches, evaluation=False):
    seq_len = min(35, len(source) - 1 - i)
    data = source[i:i+seq_len]

    if evaluation is True:
        data.detach_()

    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

import itertools
loader = itertools.starmap(get_batch, zip(range(35)))

for (batch, targets) in loader:
    print(batch.size())