import numpy as np

def load_vals(name):
    model_list = []
    for i in range(16):
        f = np.load('text_results/'+name+'_'+str(i)+'.npz')
        
        model_list.append(np.mean(f['logws']))
    
    print(model_list)
    return np.mean(model_list)

vocab_length = 35
print('VAE NLL: ', load_vals('VAE')*vocab_length)
print('BAE_G NLL: ', load_vals('BAE')*vocab_length)
print('BAE_S NLL: ', load_vals('BAE_s')*vocab_length)

print('VAE PPL: ', np.exp(load_vals('VAE')) )
print('BAE_G PPL: ', np.exp(load_vals('BAE')) )
print('BAE_S PPL: ', np.exp(load_vals('BAE_s')) )