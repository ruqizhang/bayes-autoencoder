import numpy as np

def load_and_compute_nll_ppl(name):
    f = np.load(name)
    print('Statistics for ' + name)
    print('Loss: ', f['logws'])
    print('NLL: ', f['logws'] * 35)
    print('PPL: ', np.exp(-f['logws']))

load_and_compute_nll_ppl('BAE_LSTM_seed_1.npz')
load_and_compute_nll_ppl('BAE_LSTMg_seed_1.npz')
load_and_compute_nll_ppl('VAE_LSTMg_seed_1.npz')