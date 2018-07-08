import torch
import copy

import torch.optim
import inferences

#todo: make a convert_to_onehot function

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

def save_model(epoch, model, optimizer, dir):
    print('saving model at epoch ', epoch)
    torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()},
        f = dir + '/ckpt_' + str(epoch+1) +'.pth.tar')

def calc_accuracy(model, data, y_one_hot, yvec):
    r"""
    simple function for computing accuracy
    """
    mu, logvar, logits = model(data, y_one_hot)
    _, pred = torch.max(logits, dim = 1)
    misclass = (pred.data.long() - yvec.long()).ne(int(0)).cpu().long().sum()
    return misclass.item()/data.size(0)

def make_onehot(y, num_classes=10):
    tmp = torch.zeros(y.size(0), num_classes)
    tmp[torch.arange(0, y.size(0)).long(), y.long()] = 1
    return tmp

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

def calculate_loss(model, data, K, alpha = 1, iw_function=inferences.VR, ss=False):
    priordist = torch.distributions.Normal(torch.zeros(data.size(0), model.zdim).to(model.device), torch.ones(data.size(0), model.zdim).to(model.device))
    prior_x = [priordist] * K
    z = [None] * K
    l_dist = [None] * K
    q_dist = [None] * K
    
    #pass forwards
    if not ss:
        mu, logvar = model(data)  
    else:
        mu, logvar, logits = model(data[0], data[1])  
    
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

    if not ss:
        return loss
    else:
        return logits

def calculate_ss_loss(model, data, y=None, K=1, alpha = 1, iw_function=inferences.VR, weight=300., num_classes=10):
    if y is not None:
        ul_loss, logits = calculate_loss(model, (data, y), K, alpha, iw_function)

        c_dist = torch.distributions.OneHotCategorical(logits = logits)
        c_loss = - weight * c_dist.log_prob(y)

        total_loss = ul_loss.view(-1) + c_loss
        secondary_loss = c_loss.sum()
    else:
        secondary_loss = None
        total_loss = 0.0
        for yy in range(num_classes):
            #create on-hot vector
            ycurrent = torch.zeros(data.size(0), num_classes)
            ycurrent[:,yy] = 1
            ycurrent.to(model.device)

            ul_loss, logits = calculate_loss(model, (data, ycurrent), K, alpha, iw_function)
            y_dist = torch.distributions.OneHotCategorical(logits = logits)
            y_lprob = y_dist.log_prob(ycurrent)

            total_loss += torch.exp(y_lprob) * (ul_loss.view(-1) - y_lprob)
    return total_loss.sum(), secondary_loss

def train_ss(model, optimizer, loaders, lossfn, device, epoch, log_interval, train_batches = 10, test_batches = 1, 
          cycle = True, num_classes=10):
    model.train()
    
    train_ulab_loss = 0.0
    train_lab_loss = 0.0
    train_loss = 0.0
    train_lab_loss2 = 0.0
    
    #switch if we want to do 1 epoch each or possibly many semi-supervised epochs
    #1 epoch each seems to be the standard method, but better results with cycling
    #default is cycling now
    if cycle:
        #iterator = enumerate(zip(itertools.cycle(train_loader_lab), train_loader_ulab))
        iterator = enumerate(zip(iter(loaders['lab']), iter(loaders['ulab'])))
    else:
        iterator = enumerate(itertools.zip_longest(loaders['lab'], loaders['ulab']))        
        
    for batch_idx, (lab, (data_ulab, _)) in iterator:
        batch_size = data_ulab.size(0)
        
        if batch_idx > training_data_size/batch_size:
            break
        
        if lab is not None: 
            #finish update for one-hot   
            data_lab, y = lab[0].to(device), lab[1]
            y = make_onehot(y, num_classes=num_classes)
        else:
            data_lab = None
        
        data_ulab = data_ulab.to(device)
        
        optimizer.zero_grad()
        
        if data_lab is not None:
            lab_loss, second_loss = lossfn(data_lab, y)
            lab_length = len(data_lab)
        else:
            lab_loss = torch.zeros(1, requires_grad=True).to(device)
            lab_length = 1
        
        ulab_loss,_ = lossfn(data_ulab)

        loss = lab_loss + ulab_loss
        loss.backward()
        train_loss += loss.data[0]
        
        train_ulab_loss += ulab_loss.data[0]
        if data_lab is not None:
            train_lab_loss += (lab_loss - second_loss).data[0]
        
            train_lab_loss2 += second_loss.data[0]
        
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f} {:.3f} {:.3f}'.format(
                        epoch, batch_idx * len(data_ulab), training_data_size,
                        100 * batch_size * batch_idx / training_data_size,
                        loss.data[0] / (len(data_ulab)+lab_length-1), 
                        ulab_loss.data[0]/len(data_ulab),
                        lab_loss.data[0]/lab_length))

    print('====> Epoch: {} Average loss: {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                    epoch, train_loss / (len(loaders['ulab'].dataset) + len(loaders['lab'].dataset)),
                    train_ulab_loss/len(loaders['ulab'].dataset),
                    train_lab_loss/len(loaders['lab'].dataset),
                    train_lab_loss2/len(loaders['lab'].dataset)))
    return lab_loss/len(loaders['lab'].dataset), ulab_loss/len(loaders['ulab'].dataset)
    
def test_ss(model, optimizer, loader, lossfn, device, epoch, num_classes = 10, reconstruct = False):
    model.eval()
    test_loss = 0
    misclass = 0

    for i, (data, y) in enumerate(loader):
        with torch.no_grad():  
            data = data.to(device)
            y_one_hot = make_onehot(y, num_classes=num_classes)

            
            loss,_ = lossfn(data, y_one_hot)
            test_loss += loss.data[0]
            batch_misclass = calc_accuracy(model, data, y_one_hot, y)
        misclass += data.size(0)/len(loader.dataset) * batch_misclass
        
        #save the first batch in testing
        if i == 0 and reconstruct:
            model.reconstruct_samples(data, y_one_hot)

    print('====> Test set accuracy: ', 1-misclass)
    test_loss /= len(loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
    return test_loss, 1-misclass
