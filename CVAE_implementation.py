import torch, argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from tqdm import tqdm


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='CVAE')

    # model hyper-parameter variables
    parser.add_argument('--lr', default=0.001, metavar='lr', type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=64, metavar='batch_size', type=float, help='batch_size')
    parser.add_argument('--itr', default=40, metavar='itr', type=int, help='Number of iterations')
    parser.add_argument('--latent_dim', default=20, metavar='latent_dim', type=int, help='latent dimesion')    

    args = parser.parse_args()
    
    
BATCH_SIZE = args.batch_size         
N_EPOCHS = args.itr              
LATENT_DIM = args.latent_dim           
lr = args.lr 
INPUT_DIM = 28 * 28     
HIDDEN_DIM = 256
N_CLASSES = 10         
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
transforms = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('/home/aminul/data',train=True,download=False,transform=transforms)
test_dataset = datasets.MNIST('/home/aminul/data',train=False,download=False,transform=transforms)


trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


def idx2onehot(idx, n=N_CLASSES):
    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx.data, 1)
    return onehot


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        super().__init__()

        self.linear = nn.Linear(input_dim + n_classes, hidden_dim) 
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        hidden = F.relu(self.linear(x))
        mean = self.mu(hidden)
        log_var = self.var(hidden)
        return mean, log_var

class Decoder(nn.Module):
    
    def __init__(self, latent_dim, hidden_dim, output_dim, n_classes):
        super().__init__()
 
        self.latent_to_hidden = nn.Linear(latent_dim + n_classes, hidden_dim)
        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.latent_to_hidden(x))
        generated_x = torch.sigmoid(self.hidden_to_out(x))
        
        return generated_x
    
class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        super().__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_classes)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, n_classes)

    def forward(self, x, y):

        x = torch.cat((x, y), dim=1)
        z_mu, z_var = self.encoder(x)

        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, y), dim=1)
        generated_x = self.decoder(z)

        return generated_x, z_mu, z_var

def calculate_loss(x, reconstructed_x, mean, log_var):
    RCL = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return RCL + KLD

def train():
    model.train()

    train_loss = 0
    iterator = tqdm(trainloader)
    for i, (x, y) in enumerate(iterator):
        x = x.view(-1, 28 * 28)
        x = x.to(device)

        y = idx2onehot(y.view(-1, 1))
        y = y.to(device)

        optimizer.zero_grad()
        reconstructed_x, z_mu, z_var = model(x, y)
        loss = calculate_loss(x, reconstructed_x, z_mu, z_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return train_loss

def test(epoch):
    model.eval()
    test_loss = 0
    iterator = tqdm(testloader)
    with torch.no_grad():
        for i, (x, y) in enumerate(iterator):
            x = x.view(-1, 28 * 28)
            x = x.to(device)
            
            y = idx2onehot(y.view(-1, 1))
            y = y.to(device)

            reconstructed_x, z_mu, z_var = model(x, y)
            loss = calculate_loss(x, reconstructed_x, z_mu, z_var)
            test_loss += loss.item()
            
            if i == 0:
                n = 10
                new_x = x.view(-1,1,28,28)
                new_reconstruction = reconstructed_x.view(-1,1,28,28)
                
                comparison = torch.cat([new_x[:n],
                                      new_reconstruction[:n]])
                save_image(comparison.cpu(),
                         'reconstruction_' + str(epoch) + '.png', nrow=n)
                
    return test_loss
    
def visualize_loss(loss):        
    plt.plot(loss)
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs Number of iteration")
    plt.show()
    
    
def visualize_grid(i1):
    z = torch.randn(36, LATENT_DIM).to(device)
    y1 = torch.randint(0, N_CLASSES, (36, 1)).to(dtype=torch.long)
    
    for j in range(36):
        y1[j][0] = i1

    y = idx2onehot(y1).to(device, dtype=z.dtype)  
    z = torch.cat((z, y), dim=1)
    reconstructed_img = model.decoder(z)
    
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(6, 6)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(reconstructed_img):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.detach().numpy().reshape(28, 28), cmap='Greys_r')
                if i == 35:
                    plt.savefig('Generated_' +str(i1)+'_.png', bbox_inches='tight')
                          
#visualize_loss(loss)
   
model = CVAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, N_CLASSES)
optimizer = optim.Adam(model.parameters(), lr=lr)


loss = []
for e in range(1,N_EPOCHS+1):

    train_loss = train()
    
    train_loss /= len(train_dataset)
    loss.append(train_loss)
    print(f'Epoch {e+1}, Train Loss: {train_loss:.2f}')
    
    if e%10 == 0:
        test_loss = test(e)
        
        
for i in range(0,10):
    visualize_grid(i)        