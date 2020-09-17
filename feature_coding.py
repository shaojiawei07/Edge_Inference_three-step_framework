import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
#from compression_module import *
import numpy as np
import argparse
#import resnet
import compression_resnet



parser = argparse.ArgumentParser()
#parser.add_argument('-channel', type=str, default='a', help='channel type, using \'a\' as AWGN and \'e\' as BEC ')
#parser.add_argument('-noise', type=float, default=0.1, help='channel condition')
parser.add_argument('-hid_dim', type=int, default=32, help='lens of encoded vector')
#parser.add_argument('-in_dim', type=int, default=2048, help='input dimension')
#parser.add_argument('-div_position', type=int, default=1, help='divide the layer')
#parser.add_argument('-sub_div_position', type=int, default=1, help='sub_divide the layer')
#parser.add_argument('-spatial', type=int, default=0, help='compress feature map')
parser.add_argument('-epoch', type=int, default=200, help='epoch')
parser.add_argument('-batch', type=int, default=128, help='batch size')
#parser.add_argument('-phase', type=int, default=2, help='phase = 1,2,3, means to different training phase')
parser.add_argument('-lr', type=float, default=1e-4, help='leaerning rate')
parser.add_argument('-split', type = str, default = '1', help='1,2,3,4,5')
parser.add_argument('-load', type=str)
parser.add_argument('-bit', type=int, default=8, help='bit_num')
#parser.add_argument('-hidden')
args = parser.parse_args()

#print('splitting point:',str(args.div_position)+'_'+str(args.sub_div_position),'input dim:',args.in_dim,'encoded dim',args.hid_dim,'spatial shrink:',args.spatial)
#print('channel model:',args.channel,'channel condition:',args.noise)
#print('phase:',args.phase,'epoch:',args.epoch,'batch:',args.batch,'learning rate:',args.lr)


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

#data_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   

testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform_test)
test_loader_this  = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

'''
model_dict = {
'1':compression_resnet.split_resnet1(args).to(device),
'2':compression_resnet.split_resnet2(args).to(device),
'3':compression_resnet.split_resnet3(args).to(device),
'4':compression_resnet.split_resnet4(args).to(device),
'5':compression_resnet.split_resnet5(args).to(device)
}

model = model_dict[args.split]
'''
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



        
#model = compression_resnet.split_resnet4(args).to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=5e-4)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=25, gamma=0.3)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)  
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# Start training
def train(model=model):

    best_acc = 0.93
    
    flag = 0
    #flag1= 0

    for epoch in range(num_epochs):
        if (epoch)%10 == 0:
            data_loader = torch.utils.data.DataLoader(trainset,batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
        
        
        
        
        for i, (x, y) in enumerate(data_loader):
                                  
            x = x.to(device)
            y = y.to(device)

            model.train()
            output, loss2 = model(x,flag)
            
            # Backprop and optimize
            criterion = nn.CrossEntropyLoss()
            criterion = criterion.to(device)
            #print(output.size(),y.size())
            loss1 = criterion(output, y)
            loss = loss1 + loss2
            
            if flag == 0 and loss1 < 0.2:
                flag = 1
                print('flag=1')
                #flag1 = 1
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            accuracy_result = accuracy(output,y)
            
            
            if (i+1) % int(50000/(args.batch*20)) == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, acc: {:.4f}" 
                       .format(epoch+1, num_epochs, i+1, len(data_loader), loss.item(), accuracy_result.item()))
            
            '''
            if accuracy_result > best_acc :
                best_acc = accuracy_result
                print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, acc: {:.4f}" 
                       .format(epoch+1, num_epochs, i+1, len(data_loader), loss.item(), accuracy_result.item()))
                torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracu':accuracy_result,
                    'loss':loss
                    },'epoch'+str(epoch)+'_acc{:.4f}_split4'.format(accuracy_result))
            

            
            if (epoch+1) % 10 == 0 and i == 0:
                print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, acc: {:.4f}" 
                       .format(epoch+1, num_epochs, i+1, len(data_loader), loss.item(), accuracy_result.item()))
                #torch.save(model,'ResNet18.pkl')
                torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracu':accuracy_result,
                    'loss':loss
                    },'epoch'+str(epoch)+'_acc{:.4f}_split4'.format(accuracy_result))
            '''        
        scheduler.step()        
        
        if (epoch)%1 == 0:
            output_flag = test(epoch)
            if output_flag == 1:
                print('finish')
                break
            
            
        
def test(epoch):
    with torch.no_grad():
        #model.load_state_dict(torch.load(str(args.div_position)+str(args.in_dim)+args.channel+'_'+str(args.hid_dim)+'_'+str(args.noise)+'_vgg_cifar10_vae_simple_static.pth'))
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
            #print('['+str(total)+'/10000]',(100 * correct / total))
            
        if (100 * correct / total) > 93:
                pred_best = (100 * correct / total)
                open('./final_result/acc:{:.4f}_split{}_dim{}_bit{}_'.format((100 * correct / total),args.split,args.hid_dim,args.bit)+ args.load[21:],'w').close()
                return 1
                #torch.save(model,args.channel+'_div:'+str(args.div_position)+'_sub_div:'+str(args.sub_div_position)+'_spatial:_'+str(args.spatial)+'_hid:'+str(args.hid_dim)+'_noise:'+str(args.noise)+'_acc{:.4f}_top5:{:.4f}_'.format((100 * correct / total),(100 *correct_top5/total))+'epoch:'+str(epoch)+'.pkl')
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total),'top5: {} %'.format(100* correct_top5/total))
        return 0


'''
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = 0.001 * (0.5 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''

      
if __name__=='__main__':
    train()
    #test()
