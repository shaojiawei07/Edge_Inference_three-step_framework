import torch
import torch.nn as nn
import torchvision
import torch.nn.utils.prune as prune

pretrained_resnet = torchvision.models.resnet18(pretrained = True)



class BasicBlock(nn.Module):
    """Basic Block for resnet
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    


class ResNet(nn.Module):

    def __init__(self, block, num_block=4, num_classes=10):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2_x = pretrained_resnet.layer1 #self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = pretrained_resnet.layer2 #self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = pretrained_resnet.layer3 #self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = pretrained_resnet.layer4 #self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output 

    def _prune_res_unit1(self,ratio = 0.1):
        prune.ln_structured(list(self.conv1)[0], name="weight", amount=ratio, n=1, dim=0)

    def _prune_res_unit2(self,ratio = 0.1):
        prune.ln_structured(list(self.conv2_x)[0].conv1, name="weight", amount=ratio, n=1, dim=0)
        prune.ln_structured(list(self.conv2_x)[0].conv2, name="weight", amount=ratio, n=1, dim=0)
        prune.ln_structured(list(self.conv2_x)[1].conv1, name="weight", amount=ratio, n=1, dim=0)
        prune.ln_structured(list(self.conv2_x)[1].conv2, name="weight", amount=ratio, n=1, dim=0)

    def _prune_res_unit3(self,ratio = 0.1):
        prune.ln_structured(list(self.conv3_x)[0].conv1, name="weight", amount=ratio, n=1, dim=0)
        prune.ln_structured(list(self.conv3_x)[0].conv2, name="weight", amount=ratio, n=1, dim=0)
        prune.ln_structured(list(self.conv3_x)[1].conv1, name="weight", amount=ratio, n=1, dim=0)
        prune.ln_structured(list(self.conv3_x)[1].conv2, name="weight", amount=ratio, n=1, dim=0)

    def _prune_res_unit4(self,ratio = 0.1):
        prune.ln_structured(list(self.conv4_x)[0].conv1, name="weight", amount=ratio, n=1, dim=0)
        prune.ln_structured(list(self.conv4_x)[0].conv2, name="weight", amount=ratio, n=1, dim=0)
        prune.ln_structured(list(self.conv4_x)[1].conv1, name="weight", amount=ratio, n=1, dim=0)
        prune.ln_structured(list(self.conv4_x)[1].conv2, name="weight", amount=ratio, n=1, dim=0)

    def _prune_res_unit5(self,ratio = 0.1):
        prune.ln_structured(list(self.conv5_x)[0].conv1, name="weight", amount=ratio, n=1, dim=0)
        prune.ln_structured(list(self.conv5_x)[0].conv2, name="weight", amount=ratio, n=1, dim=0)
        prune.ln_structured(list(self.conv5_x)[1].conv1, name="weight", amount=ratio, n=1, dim=0)
        prune.ln_structured(list(self.conv5_x)[1].conv2, name="weight", amount=ratio, n=1, dim=0)

    def _remove_res_unit1(self):

        list(self.conv1)[0].weight_mask.data = list(self.conv1)[0].weight_mask.data * 0 + 1

        prune.remove(list(self.conv1)[0], name="weight")

    def _remove_res_unit2(self):

        list(self.conv2_x)[0].conv1.weight_mask.data = list(self.conv2_x)[0].conv1.weight_mask.data * 0 + 1
        list(self.conv2_x)[0].conv2.weight_mask.data = list(self.conv2_x)[0].conv2.weight_mask.data * 0 + 1
        list(self.conv2_x)[1].conv1.weight_mask.data = list(self.conv2_x)[1].conv1.weight_mask.data * 0 + 1
        list(self.conv2_x)[1].conv2.weight_mask.data = list(self.conv2_x)[1].conv2.weight_mask.data * 0 + 1

        prune.remove(list(self.conv2_x)[0].conv1, name="weight")
        prune.remove(list(self.conv2_x)[0].conv2, name="weight")
        prune.remove(list(self.conv2_x)[1].conv1, name="weight")
        prune.remove(list(self.conv2_x)[1].conv2, name="weight")

    def _remove_res_unit3(self):

        list(self.conv3_x)[0].conv1.weight_mask.data = list(self.conv3_x)[0].conv1.weight_mask.data * 0 + 1
        list(self.conv3_x)[0].conv2.weight_mask.data = list(self.conv3_x)[0].conv2.weight_mask.data * 0 + 1
        list(self.conv3_x)[1].conv1.weight_mask.data = list(self.conv3_x)[1].conv1.weight_mask.data * 0 + 1
        list(self.conv3_x)[1].conv2.weight_mask.data = list(self.conv3_x)[1].conv2.weight_mask.data * 0 + 1

        prune.remove(list(self.conv3_x)[0].conv1, name="weight")
        prune.remove(list(self.conv3_x)[0].conv2, name="weight")
        prune.remove(list(self.conv3_x)[1].conv1, name="weight")
        prune.remove(list(self.conv3_x)[1].conv2, name="weight")

    def _remove_res_unit4(self):

        list(self.conv4_x)[0].conv1.weight_mask.data = list(self.conv4_x)[0].conv1.weight_mask.data * 0 + 1
        list(self.conv4_x)[0].conv2.weight_mask.data = list(self.conv4_x)[0].conv2.weight_mask.data * 0 + 1
        list(self.conv4_x)[1].conv1.weight_mask.data = list(self.conv4_x)[1].conv1.weight_mask.data * 0 + 1
        list(self.conv4_x)[1].conv2.weight_mask.data = list(self.conv4_x)[1].conv2.weight_mask.data * 0 + 1

        prune.remove(list(self.conv4_x)[0].conv1, name="weight")
        prune.remove(list(self.conv4_x)[0].conv2, name="weight")
        prune.remove(list(self.conv4_x)[1].conv1, name="weight")
        prune.remove(list(self.conv4_x)[1].conv2, name="weight")

    def _remove_res_unit5(self):

        list(self.conv5_x)[0].conv1.weight_mask.data = list(self.conv5_x)[0].conv1.weight_mask.data * 0 + 1
        list(self.conv5_x)[0].conv2.weight_mask.data = list(self.conv5_x)[0].conv2.weight_mask.data * 0 + 1
        list(self.conv5_x)[1].conv1.weight_mask.data = list(self.conv5_x)[1].conv1.weight_mask.data * 0 + 1
        list(self.conv5_x)[1].conv2.weight_mask.data = list(self.conv5_x)[1].conv2.weight_mask.data * 0 + 1

        prune.remove(list(self.conv5_x)[0].conv1, name="weight")
        prune.remove(list(self.conv5_x)[0].conv2, name="weight")
        prune.remove(list(self.conv5_x)[1].conv1, name="weight")
        prune.remove(list(self.conv5_x)[1].conv2, name="weight")



def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

if __name__ == '__main__':
    model = resnet18()
