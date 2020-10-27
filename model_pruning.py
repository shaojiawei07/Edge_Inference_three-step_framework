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
import resnet

parser = argparse.ArgumentParser()
parser.add_argument('-epoch', type=int, default=100, help='epoch')
parser.add_argument('-batch', type=int, default=128, help='batch size')
parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('-sp1', type=float, default=0.0)
parser.add_argument('-sp2', type=float, default=0.0)
parser.add_argument('-sp3', type=float, default=0.0)
parser.add_argument('-sp4', type=float, default=0.0)
parser.add_argument('-sp5', type=float, default=0.0)
args = parser.parse_args()

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
model = resnet.resnet18().to(device)

model_dict = torch.load('./demo_acc95.50')
model.load_state_dict(model_dict['model_state_dict'])

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=40, gamma=0.3)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)  
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# Start training
def train(model=model):

    prune_flag = 0

    for epoch in range(num_epochs):
        if (epoch)%10 == 0:
            data_loader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
        
        
        for i, (x, y) in enumerate(data_loader):
                                  
            x = x.to(device)
            y = y.to(device)

            if i % 150 == 0 :
                if prune_flag == 1:
                    model._remove_res_unit1()
                    model._remove_res_unit2()
                    model._remove_res_unit3()
                    model._remove_res_unit4()
                    model._remove_res_unit5()
                    prune_flag = 0

                model._prune_res_unit1(args.sp1)
                model._prune_res_unit2(args.sp2)
                model._prune_res_unit3(args.sp3)    
                model._prune_res_unit4(args.sp4)
                model._prune_res_unit5(args.sp5)
                prune_flag = 1


            model.train()
            output = model(x)
            
            criterion = nn.CrossEntropyLoss()
            criterion = criterion.to(device)
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            accuracy_result = accuracy(output,y)
            
            if (i+1) % int(50000/(args.batch*20)) == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, acc: {:.4f}" 
                       .format(epoch+1, num_epochs, i+1, len(data_loader), loss.item(), accuracy_result.item()))
        
        scheduler.step()        
        
        if (epoch)%3 == 0:
            if prune_flag == 0:
                model._prune_res_unit1(args.sp1)
                model._prune_res_unit2(args.sp2)
                model._prune_res_unit3(args.sp3)    
                model._prune_res_unit4(args.sp4)
                model._prune_res_unit5(args.sp5)
                prune_flag = 1
                
            else:
                model._remove_res_unit1()
                model._remove_res_unit2()
                model._remove_res_unit3()
                model._remove_res_unit4()
                model._remove_res_unit5()
                model._prune_res_unit1(args.sp1)
                model._prune_res_unit2(args.sp2)
                model._prune_res_unit3(args.sp3)    
                model._prune_res_unit4(args.sp4)
                model._prune_res_unit5(args.sp5)
                prune_flag = 1

            test(epoch, prune_flag)
            prune_flag = 0
            
        
def test(epoch, prune_flag):
    with torch.no_grad():

        if prune_flag == 0:
            model._prune_res_unit1(args.sp1)
            model._prune_res_unit2(args.sp2)
            model._prune_res_unit3(args.sp3)    
            model._prune_res_unit4(args.sp4)
            model._prune_res_unit5(args.sp4)

        model.eval()

        correct = 0
        correct_top5 = 0
        total = 0

        for i, (images, labels) in enumerate(test_loader_this): 
            images = images.to(device)
            labels = labels.to(device)
            outputs= model(images)
            maxk = max((1,5))
            labels_relize = labels.view(-1,1)
            _, top5_pred = outputs.topk(maxk, 1, True, True)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_top5 +=torch.eq(top5_pred, labels_relize).sum().float().item()
            correct += (predicted == labels).sum().item()
            
        model._remove_res_unit1()
        model._remove_res_unit2()
        model._remove_res_unit3()
        model._remove_res_unit4()
        model._remove_res_unit5()

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'accuracy': (100 * correct / total)
                    },'./sp1_{:.2f}_sp2_{:.2f}_sp3_{:.2f}_sp4_{:.2f}_sp5_{:.2f}_test_acc{:.4f}'.format(args.sp1,args.sp2,args.sp3,args.sp4,args.sp5,(100 * correct / total)))

        if (100 * correct / total) > 60:
                pred_best = (100 * correct / total)
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total),'top5: {} %'.format(100* correct_top5/total))
      
if __name__=='__main__':
    train()
    
