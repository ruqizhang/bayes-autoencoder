from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--device_id',type = int, help = 'device id to use')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
device_id = args.device_id
args.cuda = torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

class VAE(nn.Module):
    def __init__(self,x_dim,z_dim,hidden_dim):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, x_dim)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        xi = Variable(torch.randn(x.size()).cuda(device_id),requires_grad=False)
        h1 = self.relu(self.fc1(x+xi))
        z = self.fc21(h1)
        return z

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def prior_loss(self,std):
        prior_loss = 0.0
        for var in self.parameters():
            nn = torch.div(var, std)
            prior_loss += torch.sum(nn*nn)

        prior_loss /= datasize
        #print(prior_loss)#1e-3
        return 0.5*prior_loss

    def noise_loss(self,lr,alpha):
        noise_loss = 0.0
        # learning_rate = base_lr * np.exp(-lr_decay *min(1.0, (train_iter*args.batch_size)/float(datasize)))
        # noise_std = 2*learning_rate*alpha
        noise_std = 2*lr*alpha
        for var in self.parameters():
            if args.cuda:
                noise_loss += torch.sum(var * Variable(torch.from_numpy(np.random.normal(0, noise_std, size=var.size())).float().cuda(device_id),
                               requires_grad = False))
            else:
                noise_loss += torch.sum(var * Variable(torch.from_numpy(np.random.normal(0, noise_std, size=var.size())).float(),
                               requires_grad = False))
        noise_loss /= datasize
        #print(noise_loss)#1e-8
        return noise_loss

    def forward(self, x):
        z = self.encode(x.view(-1, self.x_dim))
        recon_x = self.decode(z)
        return recon_x,z

datasize = len(train_loader.dataset)
x_dim = 784
z_dim = 20
hidden_dim = 400
lr_decay = 3.0
base_lr = 1e-3
model = VAE(x_dim,z_dim,hidden_dim)
if args.cuda:
    model.cuda(device_id)


def loss_function(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))* 784


    return BCE

def en_loss(z_recon,z):
    z = Variable(z.data,requires_grad = False)
    loss = F.mse_loss(z_recon,z)
    return loss

def z_prior_loss(z):
    prior_loss = 0.5*torch.sum(z*z)
    return prior_loss/datasize

def z_noise_loss(z):

    learning_rate = lr

    noise_std = np.sqrt(2*learning_rate*alpha)
    noise_std = torch.from_numpy(np.array([noise_std])).float().cuda(device_id)
    noise_std = noise_std[0]
    means = torch.zeros(z.size()).cuda(device_id)
    noise_loss = torch.sum(z * Variable(torch.normal(means, std = noise_std).cuda(device_id),
                           requires_grad = False))
    #print('noise_loss',noise_loss)#1e-8
    return noise_loss/datasize
# def optimizer_fun(train_iter):
#     learning_rate = base_lr * np.exp(-lr_decay *min(1.0, (train_iter*args.batch_size)/float(datasize)))
#     opt = optim.Adam(model.parameters(), lr=learning_rate)
#     return opt
def z_opt(z_sample):

    opt = optim.SGD([z_sample], lr=lr, momentum = 1-alpha)

    return opt
def train(epoch):
    model.train()
    train_loss = 0
    count = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda(device_id)
        # optimizer = optimizer_fun(epoch*batch_idx)
        for j in range(J):
            if j == 0:
                model.zero_grad()
                recon_batch,z = model(data)
                # jacobian = compute_jacobian(xi,z)
                z_sample = Variable(z.data,requires_grad = True)
                z_optimizer = z_opt(z_sample)
                z_optimizer.zero_grad()
            else:
                recon_batch = model.decode(z_sample)

            BCE = loss_function(recon_batch, data)

            prior_loss = model.prior_loss(prior_std)
            noise_loss = model.noise_loss(lr,alpha)
            prior_loss_z = z_prior_loss(z_sample)
            noise_loss_z = z_noise_loss(z_sample)
            loss = BCE+ prior_loss+noise_loss+prior_loss_z+noise_loss_z
            if j>burnin:
                loss_en = en_loss(z_sample,z)
                loss += loss_en
            if j%2==0:
                loss.backward()
                optimizer.step()
            else:
                loss.backward()
                z_optimizer.step()

        train_loss += loss.data[0]
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: %g [%g/%g ]\tLoss: %g'%(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                loss.data[0]))
        count += 1
    print('====> Epoch: %g Average loss: %g'%(
          epoch, train_loss / count))


def test(epoch):
    model.eval()
    test_loss = 0
    count = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda(device_id)
        data = Variable(data, volatile=True)
        recon_batch,z = model(data)
        test_loss += loss_function(recon_batch, data).data[0]
        # if i == 0:
        #   n = min(data.size(0), 8)
        #   comparison = torch.cat([data[:n],
        #                           recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
        #   save_image(comparison.data.cpu(),
        #              './results_vanilla_64/reconstruction_' + str(epoch) + '.png', nrow=n)

        count += 1
    test_loss /= count
    print('====> Test set loss: %g'%(test_loss))
    return test_loss

lr = 1e-3
alpha = 0.1
prior_std = 1
burnin = 0
J = 4
best_test_nll = None
optimizer = optim.SGD(model.parameters(), lr=lr,momentum = 1-alpha)
for epoch in range(1, args.epochs + 1):
    train(epoch)
    test_nll = test(epoch)
    if not best_test_nll or test_nll < best_test_nll:
        print('save model')
        torch.save(model.state_dict(),'./results/bng_model.pt')
        best_test_nll = test_nll
model.load_state_dict(torch.load('./results/bng_model_final.pt'))
sample = Variable(torch.randn(64, z_dim))
if args.cuda:
   sample = sample.cuda(device_id)
sample = model.decode(sample).cpu()
save_image(sample.data.view(64, 1, 28, 28),
           './results/bng_sample' + str(epoch) + '.png')
