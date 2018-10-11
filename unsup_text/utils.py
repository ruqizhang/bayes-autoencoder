import torch
import copy

import torch.optim

import itertools

"""def construct_optimizer(paramlist, method, options=None):
    #create kwargs by determining what was passed in from the command line
    kwargs={}
    if options is not None:
        res = [i.split('=') for i in options]
        for k,v in res:
            try:
                kwargs[k] = float(v)
            except:
                try:
                    kwargs[k] = bool(v)
                except:
                    kwargs[k] = str(v)

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
            print('Either optimizer not found or parameters given failed')"""

def save_model(epoch, model, optimizer, dir):
    print('saving model at epoch ', epoch)
    torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()},
        f = dir + '/ckpt_' + str(epoch+1) +'.pth.tar')

"""def make_onehot(y, num_classes=10):
    tmp = torch.zeros(y.size(0), num_classes)
    tmp[torch.arange(0, y.size(0)).long(), y.long()] = 1
    return tmp

def train(model, optimizer, train_loader, lossfn, device, epoch, log_interval, **kwargs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
            
        optimizer.zero_grad()
        
        def closure():
            loss = lossfn(data, **kwargs)
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

def test(model, test_loader, lossfn, device, reconstruct=False, **kwargs):
    model.eval()
    test_loss = 0
    for i, (data, y) in enumerate(test_loader):
        data = data.to(device)

        test_loss += lossfn(data, **kwargs).item()
        
        #save the first batch in testing
        if i == 0 and reconstruct:
            model.reconstruct_samples(data, y, **kwargs)
     
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
    return test_loss


def train_ss(model, optimizer, loaders, lossfn, device, epoch, log_interval, train_batches = 10, test_batches = 1, 
          cycle = True, num_classes=10, **kwargs):
    model.train()
    
    ulab_size = len(loaders['ulab'].dataset)
    lab_size = len(loaders['lab'].dataset)
    training_data_size = ulab_size

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
            data_lab, y = lab[0].to(device), lab[1].to(device)
            y = make_onehot(y, num_classes=num_classes).to(device)
        else:
            data_lab = None
        
        data_ulab = data_ulab.to(device)
        
        optimizer.zero_grad()
        
        if data_lab is not None:
            lab_loss, second_loss = lossfn(data_lab, y, **kwargs)
            lab_length = len(data_lab)
        else:
            lab_loss = torch.zeros(1, requires_grad=True).to(device)
            lab_length = 1
        
        ulab_loss,_ = lossfn(data_ulab, **kwargs)

        loss = lab_loss + ulab_loss
        loss.backward()
        train_loss += loss.item()
        
        train_ulab_loss += ulab_loss.item()
        if data_lab is not None:
            train_lab_loss += (lab_loss - second_loss).item()
        
            train_lab_loss2 += second_loss.item()
        
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f} {:.3f} {:.3f}'.format(
                        epoch, batch_idx * len(data_ulab), training_data_size,
                        100 * batch_size * batch_idx / training_data_size,
                        loss.item() / (len(data_ulab)+lab_length-1), 
                        ulab_loss.item()/len(data_ulab),
                        lab_loss.item()/lab_length))

    print('====> Epoch: {} Average loss: {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                    epoch, train_loss / (ulab_size + lab_size),
                    train_ulab_loss/ulab_size,
                    train_lab_loss/lab_size,
                    train_lab_loss2/lab_size))
    return lab_loss/lab_size, ulab_loss/ulab_size
    
def test_ss(model, loader, lossfn, calc_accuracy, device, epoch, num_classes = 10, reconstruct = False, **kwargs):
    model.eval()
    test_loss = 0
    misclass = 0.0

    for i, (data, y) in enumerate(loader):
        with torch.no_grad():  
            data = data.to(device)
            y = y.to(device)
            y_one_hot = make_onehot(y, num_classes=num_classes).to(device)

            loss,_ = lossfn(data, y_one_hot, **kwargs)
            test_loss += loss.item()
            batch_misclass = calc_accuracy(data, y_one_hot, y)
        misclass += data.size(0)/len(loader.dataset) * batch_misclass
        #save the first batch in testing
        #if i == 0 and reconstruct:
        #    model.reconstruct_samples(data, y_one_hot)

    print('====> Test set accuracy: ', 1-misclass)
    test_loss /= len(loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
    return test_loss, 1-misclass"""
