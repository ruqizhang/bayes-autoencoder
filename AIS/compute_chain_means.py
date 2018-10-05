import numpy as np

def load_vals(name):
    model_list = []
    for i in range(16):
        f = np.load('text_results/'+name+'_'+str(i)+'.npz')
        model_list.append(np.mean(f['logws']))
    
    return np.mean(model_list)

vocab_length = 10000.
print('VAE LL: ', load_vals('VAE')/vocab_length)
print('BAE_G LL: ', load_vals('BAE')/vocab_length)
print('BAE_S LL: ', load_vals('BAE_s')/vocab_length)