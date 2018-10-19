import numpy as np

def load_vals(name):
    model_list = []
    for i in range(16):
        f = np.load('old_text_results/'+name+'_'+str(i)+'.npz')
        print(len(f['logws']))
        model_list.append(np.mean(f['logws']))
    
    print(model_list)
    return np.mean(model_list)

vocab_length = 10000.
print('VAE LL: ', load_vals('VAE')/vocab_length)
print('BAE_G LL: ', load_vals('BAE')/vocab_length)
print('BAE_S LL: ', load_vals('BAE_s')/vocab_length)