import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
import numpy as np
import argparse
import compression_resnet



parser = argparse.ArgumentParser()
parser.add_argument('-hid_dim', type=int, default=32, help='dimension of encoded vector')
parser.add_argument('-epoch', type=int, default=60, help='epoch')
parser.add_argument('-batch', type=int, default=128, help='batch size')
parser.add_argument('-lr', type=float, default=1e-4, help='leaerning rate')
parser.add_argument('-split', type = str, default = '1', help='1,2,3,4,5')
parser.add_argument('-load', type=str)
parser.add_argument('-bit', type=int, default=8, help='bit_num')
args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = args.epoch
batch_size = args.batch
learning_rate = args.lr

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform_test)
test_loader_this  = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# select the split point
# at the end of the i-th building block in ResNet
if args.split == '1':
    model = compression_resnet.split_resnet1(args).to(device)
    print('1')
elif args.split == '2':
    model = compression_resnet.split_resnet2(args).to(device)
    print('2')
elif args.split == '3':
    model = compression_resnet.split_resnet3(args).to(device)
    print('3')
elif args.split == '4':
    model = compression_resnet.split_resnet4(args).to(device)
    print('4')
elif args.split == '5':
    model = compression_resnet.split_resnet5(args).to(device)
    print('5')


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=25, gamma=0.3)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)  
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# Start training
def train(model=model):
    
    flag = 0

    for epoch in range(num_epochs):
        if (epoch)%10 == 0:
            data_loader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
        
        for i, (x, y) in enumerate(data_loader):
                                  
            x = x.to(device)
            y = y.to(device)

            model.train()
            output, loss2 = model(x,flag)
            
            criterion = nn.CrossEntropyLoss()
            criterion = criterion.to(device)
            loss1 = criterion(output, y)
            loss = loss1 + loss2
            
            #warmup process until loss1 < 0.2
            if flag == 0 and loss1 < 0.2:
                flag = 1
                print('flag=1')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            accuracy_result = accuracy(output,y)
            
            
            if (i+1) % int(50000/(args.batch*20)) == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, acc: {:.4f}" 
                       .format(epoch+1, num_epochs, i+1, len(data_loader), loss.item(), accuracy_result.item()))        
        
        scheduler.step()                
        test(epoch)
            
            
        
def test(epoch):
    with torch.no_grad():
        
        model.eval()

        correct = 0
        correct_top5 = 0#top5
        total = 0

        for i, (images, labels) in enumerate(test_loader_this): 
            images = images.to(device)
            labels = labels.to(device)
            outputs,_= model(images,1)
            maxk = max((1,5))
            labels_relize = labels.view(-1,1)
            _, top5_pred = outputs.topk(maxk, 1, True, True)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_top5 +=torch.eq(top5_pred, labels_relize).sum().float().item()
            correct += (predicted == labels).sum().item()
                            
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total),'top5: {} %'.format(100* correct_top5/total))


      
if __name__=='__main__':
    train()
