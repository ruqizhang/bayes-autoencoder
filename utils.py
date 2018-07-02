import torch
import copy

import torch.optim
import inferences

def construct_optimizer(paramlist, method, options=None):
    #create kwargs by determining what was passed in from the command line
    kwargs={}
    if options is not None:
        res = [i.split('=') for i in options]
        for k,v in res:
            kwargs[k] = float(v)

    #create and return optimizer
    #sghmc is different
    if method == 'SGHMC':
        print('Using SGHMC - please remember that this is sampling, not MLE training.\n Parameters are:')
        from sghmc import SGHMC
        optimizer = SGHMC(paramlist, **kwargs)
        print(optimizer.defaults)
        return optimizer
    #create the optimizer by look-up
    else:
        all_optimizers = copy.deepcopy(torch.optim.Optimizer.__subclasses__())
        match = [i for i, s in enumerate(str(all_optimizers).split('>')) if '.'+ method +'\'' in s]

        try:
            print('Using', method, 'with following parameters:')
            optimizer = all_optimizers[match[0]](paramlist, **kwargs)
            print(optimizer.defaults)
            return optimizer
        except:
            print('Either optimizer not found or parameters given failed')

def train(model, optimizer, train_loader, lossfn, device, epoch, log_interval):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
            
        optimizer.zero_grad()
        
        def closure():
            loss = lossfn(data)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        
        train_loss += loss.item()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    
    return train_loss/len(train_loader.dataset)

def test(model, test_loader, lossfn, K, device, reconstruct=False, **kwargs):
    model.eval()
    test_loss = 0
    for i, (data, y) in enumerate(test_loader):
        data = data.to(device)

        test_loss += lossfn(data, K).item()
        
        #save the first batch in testing
        if i == 0 and reconstruct:
            model.reconstruct_samples(data, y, **kwargs)
     
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
    return test_loss

def save_model(epoch, model, optimizer, dir):
    print('saving model at epoch ', epoch)
    torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()},
        f = dir + '/ckpt_' + str(epoch+1) +'.pth.tar')

def calculate_loss(model, data, K, alpha = 1, iw_function=inferences.VR):
    priordist = torch.distributions.Normal(torch.zeros(data.size(0), model.zdim).to(model.device), torch.ones(data.size(0), model.zdim).to(model.device))
    prior_x = [priordist] * K
    z = [None] * K
    l_dist = [None] * K
    q_dist = [None] * K
    
    #pass forwards
    mu, logvar = model(data)    
    
    for k in range(K):
        #generate distribution
        q_dist[k] = torch.distributions.Normal(mu, torch.exp(logvar.mul(0.5)))
        
        #reparameterize
        z[k] = q_dist[k].rsample()
    
        #pass backwards
        x_probs = model.decode(z[k])
    
        #create distributions
        l_dist[k] = torch.distributions.Bernoulli(probs = x_probs)
        
    #compute loss function
    loss = iw_function(z, data.view(-1,784), l_dist, prior_x, q_dist, K = K, alpha = alpha)
    return loss