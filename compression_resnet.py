import torch
import torch.nn as nn
import torchvision
import torch.nn.utils.prune as prune
import resnet


class emb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, embb):

        x_expanded = input.unsqueeze(-1)
        emb_expanded = embb
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        emb_t = embb.t()
        x = emb_t[argmin,:]

        return x,argmin

    @staticmethod
    def backward(ctx, grad_output, argmin = None):
        grad_input = grad_output.clone()
        return grad_input, None


class split_resnet1(nn.Module):
    def __init__(self,hidden_channel = 128):
        super().__init__()
        self.hidden_channel = hidden_channel

        self.resnet = resnet.resnet18()
        model_dict = torch.load('./target_epoch499_acc1.0000')
        self.resnet.load_state_dict(model_dict['model_state_dict'])

        self.encode1 = nn.Linear(32768*2 ,self.hidden_channel)
        self.encode2 = nn.Linear(self.hidden_channel,self.hidden_channel)
        self.decode1 = nn.Linear(self.hidden_channel,self.hidden_channel)
        self.decode2 = nn.Linear(self.hidden_channel,32768*2)

        self.elu = nn.ELU()
        self.Tanh = nn.Tanh()

        #codeword
        self.weight = nn.Parameter(torch.rand(hidden_channel,256))

        for para in self.resnet.parameters():
            para.requires_grad = False


    def forward(self, x, flag):
        output = self.resnet.conv1(x)

        B = output.size()[0]

        output = torch.reshape(output,(B,-1))

        output = self.elu(self.encode1(output))
        output = self.Tanh(self.encode2(output))

        output_tmp = output

        if flag == 1:
            output_emb, argmin = emb().apply(output, self.weight.detach())
            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach()) + 0.5 * F.mse_loss(selected_emb.detach(),output_tmp)

        elif flag == 0:
            output_emb = output_tmp
            _, argmin = emb().apply(output self.weight.detach())
            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach())

        output = self.elu(self.decode1(output))
        output = self.elu(self.decode2(output))

        output = torch.reshape(output,(B,64,32,32))
        
        output = self.resnet.conv2_x(output)

        output = self.resnet.conv3_x(output)
        output = self.resnet.conv4_x(output)
        output = self.resnet.conv5_x(output)
        output = self.resnet.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.resnet.fc(output)
        
        return output, regular_term


class split_resnet2(nn.Module):
    def __init__(self,hidden_channel = 128):
        super().__init__()
        self.hidden_channel = hidden_channel

        self.resnet = resnet.resnet18()
        model_dict = torch.load('./target_epoch499_acc1.0000')
        self.resnet.load_state_dict(model_dict['model_state_dict'])

        self.encode1 = nn.Linear(32768*2 ,self.hidden_channel)
        self.encode2 = nn.Linear(self.hidden_channel,self.hidden_channel)
        self.decode1 = nn.Linear(self.hidden_channel,self.hidden_channel)
        self.decode2 = nn.Linear(self.hidden_channel,32768*2)

        #self.enc_dec = nn.sequanen

        #nn.BatchNorm2d

        self.elu = nn.ELU()
        self.Tanh = nn.Tanh()
        self.weight = nn.Parameter(torch.rand(hidden_channel,256))

        for para in self.resnet.parameters():
            para.requires_grad = False


    def forward(self, x, flag):
        output = self.resnet.conv1(x)
        output = self.resnet.conv2_x(output)

        B = output.size()[0]
        #print(output.size())
        output = torch.reshape(output,(B,-1))

        output = self.elu(self.encode1(output))
        output = self.Tanh(self.encode2(output))

        output_tmp = output

        if flag == 1:
            output_emb, argmin = emb().apply(output, self.weight.detach())

            #selected_emb = emb().apply(x.detach(), self.weight)

            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach()) + 0.5 * F.mse_loss(selected_emb.detach(),output_tmp)

        elif flag == 0:
            output_emb = output_tmp
            _, argmin = emb().apply(output self.weight.detach())
            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach())


        #emb().apply(output, self.weight.detach())


        output = self.elu(self.decode1(output))
        output = self.elu(self.decode2(output))

        output = torch.reshape(output,(B,64,32,32))

        output = self.resnet.conv3_x(output)
        output = self.resnet.conv4_x(output)
        output = self.resnet.conv5_x(output)
        output = self.resnet.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.resnet.fc(output)
        
        return output, regular_term


class split_resnet3(nn.Module):
    def __init__(self,hidden_channel = 128):
        super().__init__()
        self.hidden_channel = hidden_channel

        self.resnet = resnet.resnet18()
        model_dict = torch.load('./target_epoch499_acc1.0000')
        self.resnet.load_state_dict(model_dict['model_state_dict'])

        self.encode1 = nn.Linear(32768,self.hidden_channel)
        self.encode2 = nn.Linear(self.hidden_channel,self.hidden_channel)
        self.decode1 = nn.Linear(self.hidden_channel,self.hidden_channel)
        self.decode2 = nn.Linear(self.hidden_channel,32768)

        for para in self.resnet.parameters():
            para.requires_grad = False

        #self.enc_dec = nn.sequanen

        #nn.BatchNorm2d

        self.elu = nn.ELU()
        self.Tanh = nn.Tanh()
        self.weight = nn.Parameter(torch.rand(hidden_channel,256))


    def forward(self, x, flag):
        output = self.resnet.conv1(x)
        output = self.resnet.conv2_x(output)
        output = self.resnet.conv3_x(output)


        B = output.size()[0]
        #print(output.size())
        output = torch.reshape(output,(B,-1))

        output = self.elu(self.encode1(output))
        output = self.Tanh(self.encode2(output))

        output_tmp = output

        if flag == 1:
            output_emb, argmin = emb().apply(output, self.weight.detach())

            #selected_emb = emb().apply(x.detach(), self.weight)

            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach()) + 0.5 * F.mse_loss(selected_emb.detach(),output_tmp)

        elif flag == 0:
            output_emb = output_tmp
            _, argmin = emb().apply(output self.weight.detach())
            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach())


        #emb().apply(output, self.weight.detach())


        output = self.elu(self.decode1(output))
        output = self.elu(self.decode2(output))

        output = torch.reshape(output,(B,128,16,16))


        output = self.resnet.conv4_x(output)

        

        output = self.resnet.conv5_x(output)
        output = self.resnet.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.resnet.fc(output)

        for para in self.resnet.parameters():
            para.requires_grad = False
        
        return output, regular_term

class split_resnet4(nn.Module):
    def __init__(self,hidden_channel = 128):
        super().__init__()
        self.hidden_channel = hidden_channel

        self.resnet = resnet.resnet18()
        model_dict = torch.load('./target_epoch499_acc1.0000')
        self.resnet.load_state_dict(model_dict['model_state_dict'])

        self.encode1 = nn.Linear(16384,self.hidden_channel)
        self.encode2 = nn.Linear(self.hidden_channel,self.hidden_channel)
        self.decode1 = nn.Linear(self.hidden_channel,self.hidden_channel)
        self.decode2 = nn.Linear(self.hidden_channel,16384)

        #self.enc_dec = nn.sequanen

        #nn.BatchNorm2d

        self.elu = nn.ELU()
        self.Tanh = nn.Tanh()
        self.weight = nn.Parameter(torch.rand(hidden_channel,256))

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
        output = self.Tanh(self.encode2(output))

        output_tmp = output

        if flag == 1:
            output_emb, argmin = emb().apply(output, self.weight.detach())

            #selected_emb = emb().apply(x.detach(), self.weight)

            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach()) + 0.5 * F.mse_loss(selected_emb.detach(),output_tmp)

        elif flag == 0:
            output_emb = output_tmp
            _, argmin = emb().apply(output self.weight.detach())
            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach())


        #emb().apply(output, self.weight.detach())


        output = self.elu(self.decode1(output))
        output = self.elu(self.decode2(output))

        output = torch.reshape(output,(B,256,8,8))

        output = self.resnet.conv5_x(output)
        output = self.resnet.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.resnet.fc(output)
        
        return output, regular_term

class split_resnet5(nn.Module):
    def __init__(self,hidden_channel = 128):
        super().__init__()
        self.hidden_channel = hidden_channel

        self.resnet = resnet.resnet18()
        model_dict = torch.load('./target_epoch499_acc1.0000')
        self.resnet.load_state_dict(model_dict['model_state_dict'])

        self.encode1 = nn.Linear(512,self.hidden_channel)
        self.encode2 = nn.Linear(self.hidden_channel,self.hidden_channel)
        self.decode1 = nn.Linear(self.hidden_channel,self.hidden_channel)
        self.decode2 = nn.Linear(self.hidden_channel,512)


        self.elu = nn.ELU()
        self.Tanh = nn.Tanh()
        self.weight = nn.Parameter(torch.rand(hidden_channel,256))

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

        output = self.elu(self.encode1(output))
        output = self.Tanh(self.encode2(output))

        output_tmp = output

        if flag == 1:
            output_emb, argmin = emb().apply(output, self.weight.detach())

            #selected_emb = emb().apply(x.detach(), self.weight)

            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach()) + 0.5 * F.mse_loss(selected_emb.detach(),output_tmp)

        elif flag == 0:
            output_emb = output_tmp
            _, argmin = emb().apply(output self.weight.detach())
            selected_emb = self.weight.t()[argmin.detach(),:]
            regular_term = F.mse_loss(selected_emb,output_tmp.detach())




        #emb().apply(output, self.weight.detach())


        output = self.elu(self.decode1(output))
        output = self.elu(self.decode2(output))

        output = self.resnet.fc(output)
        
        return output, regular_term
