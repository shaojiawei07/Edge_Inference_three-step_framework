import torch
import torch.nn as nn
import torchvision
import torch.nn.utils.prune as prune
import resnet
import torch.nn.functional as F


class Round(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        #ctx.save_for_backward(input)
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        #input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        #grad_input[input < 0] = 0
        return grad_input

class emb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, embb):

        x_expanded = input.unsqueeze(-1)
        emb_expanded = embb
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        #shifted_shape = [input.shape[0], *list(input.shape[2:]) ,input.shape[1]]
        #result = emb.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        emb_t = embb.t()

        x = emb_t[argmin,:]

        #ctx.save_for_backward(argmin)
        return x,argmin

    @staticmethod
    def backward(ctx, grad_output, argmin = None):
        grad_input = grad_output.clone()
        return grad_input, None


class split_resnet1(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.hidden_channel = args.hid_dim
        self.scale = 2 ** args.bit

        self.resnet = resnet.resnet18()
        model_dict = torch.load(args.load)
        self.resnet.load_state_dict(model_dict['model_state_dict'])

        #self.encode1 = nn.Linear(32768*2 ,self.hidden_channel)
        #self.encode2 = nn.Linear(self.hidden_channel,self.hidden_channel)
        #self.decode1 = nn.Linear(self.hidden_channel,self.hidden_channel)
        #self.decode2 = nn.Linear(self.hidden_channel,32768*2)
        
        self.conv1 = nn.Conv2d(64,self.hidden_channel,kernel_size = 3,stride=1,padding=1)
        #self.conv1_1 = nn.Conv2d(64,32,kernel_size = 3,stride=1,padding=1)
        
        self.conv2 = nn.Conv2d(self.hidden_channel,64,kernel_size = 3,stride=1,padding=1)
        #self.conv2_1 = nn.Conv2d(32,64,kernel_size = 3,stride=1,padding=1)

        self.elu = nn.ELU()
        self.Tanh = nn.Tanh()
        self.weight = nn.Parameter(torch.rand(self.hidden_channel* 32 * 32 ,2**args.bit))

        for para in self.resnet.parameters():
            para.requires_grad = False


    def forward(self, x, flag):
        output = self.resnet.conv1(x)

        B = output.size()[0]
        scale = self.scale
        #print(output.size())
        #output = torch.reshape(output,(B,-1))

        #output = self.elu(self.encode1(output))
        output = self.Tanh(self.conv1(output))

        output_ori = torch.reshape(output,(B,-1))

        output = Round().apply(scale*(output_ori + 1)/2.0)

        output = (output * 2)/scale - 1

        regular_term = torch.norm(output-output_ori)**2 / B
        
        #output = torch.reshape(output,(B,-1))
        
        '''

        output_tmp = output

        if flag == 1:
            output_emb, argmin = emb().apply(output, self.weight.detach())

            #selected_emb = emb().apply(x.detach(), self.weight)

            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach()) + 0.5 * F.mse_loss(selected_emb.detach(),output_tmp)

        elif flag == 0:
            output_emb = output_tmp
            _, argmin = emb().apply(output, self.weight.detach())
            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach())
        


        output = torch.reshape(output,(B,self.hidden_channel,32,32))
        '''
        #emb().apply(output, self.weight.detach())
        output = torch.reshape(output,(B,self.hidden_channel,32,32))


        #output = self.elu(self.decode1(output))
        output = self.elu(self.conv2(output))

        #output = torch.reshape(output,(B,64,32,32))
        
        output = self.resnet.conv2_x(output)

        output = self.resnet.conv3_x(output)
        output = self.resnet.conv4_x(output)
        output = self.resnet.conv5_x(output)
        output = self.resnet.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.resnet.fc(output)
        
        return output, regular_term


class split_resnet2(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.hidden_channel = args.hid_dim

        self.resnet = resnet.resnet18()
        model_dict = torch.load(args.load)
        self.resnet.load_state_dict(model_dict['model_state_dict'])
        self.scale = 2 ** args.bit
        print('scale',self.scale,'bit',args.bit)

        #self.encode1 = nn.Linear(32768*2 ,self.hidden_channel)
        #self.encode2 = nn.Linear(self.hidden_channel,self.hidden_channel)
        #self.decode1 = nn.Linear(self.hidden_channel,self.hidden_channel)
        #self.decode2 = nn.Linear(self.hidden_channel,32768*2)

        #self.enc_dec = nn.sequanen

        #nn.BatchNorm2d
        
        self.conv1 = nn.Conv2d(64,self.hidden_channel,kernel_size = 3,stride=1,padding=1)
        #self.conv1_1 = nn.Conv2d(64,32,kernel_size = 3,stride=1,padding=1)
        
        self.conv2 = nn.Conv2d(self.hidden_channel,64,kernel_size = 3,stride=1,padding=1)

        self.elu = nn.ELU()
        self.Tanh = nn.Tanh()
        self.weight = nn.Parameter(torch.rand(self.hidden_channel,2**args.bit))

        for para in self.resnet.parameters():
            para.requires_grad = False


    def forward(self, x, flag):
        output = self.resnet.conv1(x)
        output = self.resnet.conv2_x(output)

        B = output.size()[0]
        #print(output.size())
        scale = self.scale
        #print(output.size())
        #output = torch.reshape(output,(B,-1))

        #output = self.elu(self.encode1(output))
        output = self.Tanh(self.conv1(output))

        output_ori = torch.reshape(output,(B,-1))

        output = Round().apply(scale*(output_ori + 1)/2.0)

        output = (output * 2)/scale - 1

        regular_term = torch.norm(output-output_ori)**2 / B


        
        #output = torch.reshape(output,(B,-1))
        
        '''

        output_tmp = output

        if flag == 1:
            output_emb, argmin = emb().apply(output, self.weight.detach())

            #selected_emb = emb().apply(x.detach(), self.weight)

            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach()) + 0.5 * F.mse_loss(selected_emb.detach(),output_tmp)

        elif flag == 0:
            output_emb = output_tmp
            _, argmin = emb().apply(output, self.weight.detach())
            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach())
        


        output = torch.reshape(output,(B,self.hidden_channel,32,32))
        '''
        #emb().apply(output, self.weight.detach())


        #output = self.elu(self.decode1(output))
        output = torch.reshape(output,(B,self.hidden_channel,32,32))
        output = self.elu(self.conv2(output))

        output = self.resnet.conv3_x(output)
        output = self.resnet.conv4_x(output)
        output = self.resnet.conv5_x(output)
        output = self.resnet.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.resnet.fc(output)
        
        return output, regular_term


class split_resnet3(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.hidden_channel = args.hid_dim
        self.scale = 2 ** args.bit

        self.resnet = resnet.resnet18()
        model_dict = torch.load(args.load)
        self.resnet.load_state_dict(model_dict['model_state_dict'])
        
        
        self.conv1 = nn.Conv2d(128,self.hidden_channel,kernel_size = 3,stride=1,padding=1)
        #self.conv1_1 = nn.Conv2d(self.hidden_channel,32,kernel_size = 3,stride=1,padding=1)
        
        self.conv2 = nn.Conv2d(self.hidden_channel,128,kernel_size = 3,stride=1,padding=1)
        #self.conv2_1 = nn.Conv2d(32,64,kernel_size = 3,stride=1,padding=1)

        
        #self.encode1 = nn.Linear(int(32768/4),self.hidden_channel)
        #self.encode2 = nn.Linear(self.hidden_channel,self.hidden_channel)
        #self.decode1 = nn.Linear(self.hidden_channel,self.hidden_channel)
        #self.decode2 = nn.Linear(self.hidden_channel,int(32768/4))
        
        #self.batchnorm1 = nn.BatchNorm2d(64)
        #self.batchnorm2 = nn.BatchNorm2d(32)
        #self.batchnorm3 = nn.BatchNorm2d(64)
        #self.batchnorm4 = nn.BatchNorm2d(128)
        
        #self.dropout = torch.nn.Dropout(p=0.25)

        for para in self.resnet.parameters():
            para.requires_grad = False

        #self.enc_dec = nn.sequanen

        #nn.BatchNorm2d

        self.elu = nn.ELU()
        self.Tanh = nn.Tanh()
        #self.weight = nn.Parameter(torch.rand(self.hidden_channel,2**args.bit))


    def forward(self, x, flag):
        output = self.resnet.conv1(x)
        output = self.resnet.conv2_x(output)
        output = self.resnet.conv3_x(output)

        B = output.size()[0]
        #print(output.size())
        scale = self.scale
        #print(output.size())
        #output = torch.reshape(output,(B,-1))

        #output = self.elu(self.encode1(output))
        output = self.Tanh(self.conv1(output))

        output_ori = torch.reshape(output,(B,-1))

        output = Round().apply(scale*(output_ori + 1)/2.0)

        output = (output * 2)/scale - 1

        regular_term = torch.norm(output-output_ori)**2 / B


        
        #output = torch.reshape(output,(B,-1))
        
        '''

        output_tmp = output

        if flag == 1:
            output_emb, argmin = emb().apply(output, self.weight.detach())

            #selected_emb = emb().apply(x.detach(), self.weight)

            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach()) + 0.5 * F.mse_loss(selected_emb.detach(),output_tmp)

        elif flag == 0:
            output_emb = output_tmp
            _, argmin = emb().apply(output, self.weight.detach())
            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach())
        


        output = torch.reshape(output,(B,self.hidden_channel,32,32))
        '''
        #emb().apply(output, self.weight.detach())


        #output = self.elu(self.decode1(output))
        output = torch.reshape(output,(B,self.hidden_channel,16,16))
        output = self.elu(self.conv2(output))

        
        output = self.resnet.conv4_x(output)
        output = self.resnet.conv5_x(output)
        output = self.resnet.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.resnet.fc(output)
        
        return output, regular_term

class split_resnet4(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.hidden_channel = args.hid_dim

        self.resnet = resnet.resnet18()
        model_dict = torch.load(args.load)
        self.resnet.load_state_dict(model_dict['model_state_dict'])

        self.encode1 = nn.Linear(16384,self.hidden_channel)
        self.encode2 = nn.Linear(self.hidden_channel,self.hidden_channel)
        self.decode1 = nn.Linear(self.hidden_channel,self.hidden_channel)
        self.decode2 = nn.Linear(self.hidden_channel,16384)
        self.dropout = torch.nn.Dropout(p = 0.25)

        #self.conv1 = nn.Conv2d(256,8,kernel_size = 3,stride=1,padding=1)
        #self.conv1_1 = nn.Conv2d(self.hidden_channel,32,kernel_size = 3,stride=1,padding=1)
        
        #self.conv2 = nn.Conv2d(8,256,kernel_size = 3,stride=1,padding=1)

        #self.enc_dec = nn.sequanen

        #nn.BatchNorm2d

        self.elu = nn.ELU()
        self.Tanh = nn.Tanh()
        self.weight = nn.Parameter(torch.rand(self.hidden_channel,2**args.bit))

        self.batchnorm1 = nn.BatchNorm2d(8)
        self.batchnorm2 = nn.BatchNorm2d(256)

        for para in self.resnet.parameters():
            para.requires_grad = False

    def forward(self, x, flag):
        output = self.resnet.conv1(x)
        output = self.resnet.conv2_x(output)
        output = self.resnet.conv3_x(output)
        output = self.resnet.conv4_x(output)

        B = output.size()[0]
        #print(output.size())
        output = torch.reshape(output,(B,-1))

        output = self.elu(self.encode1(output))
        output = self.dropout(output)
        output = self.Tanh(self.encode2(output))
        
        

        output_tmp = output

        if flag == 1:
            output_emb, argmin = emb().apply(output, self.weight.detach())

            #selected_emb = emb().apply(x.detach(), self.weight)

            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach()) + 0.5 * F.mse_loss(selected_emb.detach(),output_tmp)

        elif flag == 0:
            output_emb = output_tmp
            _, argmin = emb().apply(output, self.weight.detach())
            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach())


        #emb().apply(output, self.weight.detach())


        output = self.elu(self.decode1(output))
        output = self.dropout(output)
        output = self.elu(self.decode2(output))

        output = torch.reshape(output,(B,256,8,8))

        output = self.resnet.conv5_x(output)
        output = self.resnet.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.resnet.fc(output)
        
        return output, 0.1 * regular_term


class split_resnet5(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.hidden_channel = args.hid_dim

        self.resnet = resnet.resnet18()
        model_dict = torch.load(args.load)
        self.resnet.load_state_dict(model_dict['model_state_dict'])

        self.encode1 = nn.Linear(512,self.hidden_channel)
        #self.encode2 = nn.Linear(self.hidden_channel,self.hidden_channel)
        #self.decode1 = nn.Linear(self.hidden_channel,self.hidden_channel)
        self.decode2 = nn.Linear(self.hidden_channel,512)


        self.elu = nn.ELU()
        self.Tanh = nn.Tanh()
        self.weight = nn.Parameter(torch.rand(self.hidden_channel,2**args.bit))

        for para in self.resnet.parameters():
            para.requires_grad = False


    def forward(self, x, flag):
        output = self.resnet.conv1(x)
        output = self.resnet.conv2_x(output)
        output = self.resnet.conv3_x(output)
        output = self.resnet.conv4_x(output)

        B = output.size()[0]
        #print(output.size())
        #output = torch.reshape(output,(B,-1))

        #output = torch.reshape(output,(B,256,8,8))

        output = self.resnet.conv5_x(output)
        output = self.resnet.avg_pool(output)
        output = output.view(output.size(0), -1)

        output = self.Tanh(self.encode1(output))
        #output = self.Tanh(self.encode2(output))

        output_tmp = output

        if flag == 1:
            output_emb, argmin = emb().apply(output, self.weight.detach())

            #selected_emb = emb().apply(x.detach(), self.weight)

            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach()) + 0.5 * F.mse_loss(selected_emb.detach(),output_tmp)

        elif flag == 0:
            output_emb = output_tmp
            _, argmin = emb().apply(output, self.weight.detach())
            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach())




        #emb().apply(output, self.weight.detach())


        #output = self.elu(self.decode1(output))
        output = self.elu(self.decode2(output))

        output = self.resnet.fc(output)
        
        return output, regular_term
