""" All layers that required for model"""

import torch
import torch.nn as nn
import torchvision
from srn.util import Singleton, init_kaiming, init_conv_gaussian

class Resnet152(nn.Module, metaclass=Singleton):
    """
    Load Resnet152 that wrapped by Singleton to make sure that the
    initiating process run only once in both F_CNN and F_Classification.
    """
    def __init__(self):
        super(Resnet152, self).__init__()
        self.model = torchvision.models.resnet152(pretrained=True)

    def forward(self, x):
        return self.model(x)

class F_CNN(nn.Module):
    """ Given an image, we use Resnet152 to extract the feature map
    We now temporarily freeze all layer in resnet 152

    Shape:
        - input x: (batch_size, 3, 224, 224)
        - output: (batch_size, 1024, 14, 14)
    """
    def __init__(self):
        super(F_CNN, self).__init__()
        resnet152 = Resnet152().model
        self.model = nn.Sequential(*list(resnet152.children())[:-3])

        # Freeze all layer
        for _, param in self.model.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)

class F_Classification(nn.Module):
    """ Res-2048 stands for one ResNet building block with 2048 output channels
    There are 3 blocks in Classification layer, the last 3 layers in Resnet152.
    Also, we unfreeze which allow the backpropagation through network.
    """
    def __init__(self, num_classes=10):
        super(F_Classification, self).__init__()
        resnet152 = Resnet152().model
        self.model = nn.Sequential(*list(resnet152.children())[-3:-1])
        self.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=num_classes)
        )
        self.fc.apply(init_kaiming)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class F_Attention(nn.Module):
    """ Given the features map extracted from the first 3 layers of Resnet,
    the function needs to compute the Attention Map: A. To make sure the
    output shape has the W and H constantly get the value of 14 given
    different value of input W and H, we use AdaptiveAveragePooling layer.

    Follow the paper, the attention estimator `f_att(.)` is modeled as 3
    convolution layers with 512 kernels of 1 x 1, 512 kernels of 3 x 3, and
    C kernels of 1 x 1, respectively. The ReLU nonlinearity operations are
    performed following the first 2 convolution layers

    Shape:
        - input x: (batch_size, 1024, H, W)
        - output: (batch_size, num_classes, 14, 14)

        Please note that the input image should be resize `width` or
        `height` to 224 px. Therefore, H or W in the above input x either
        will get the value of 14. For example, the valid input x should be:
        [batch_size, 1024, 14, 16] or [batch_size, 1024, 21, 14], shouldn't
        be: [batch_size, 1024, 21, 20].

    """
    def __init__(self, num_classes=10):
        super(F_Attention, self).__init__()
        self.num_classes = num_classes
        self.adap_average_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.residual_att_1 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.residual_att_2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.att = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, bias=True)
        )
        self.residual_att_1.apply(init_conv_gaussian)
        self.residual_att_2.apply(init_conv_gaussian)
        self.att.apply(lambda x: init_conv_gaussian(x, fill_bias=0))

    def forward(self, x):
        assert x.shape[1] == 1024
        x = self.adap_average_pool(x)

        out1 = self.residual_att_1(x)
        x = x + out1
        out2 = self.residual_att_2(x)
        x = x + out2

        x = self.att(x)

        # softmax over W and H.
        assert x.shape[1] == self.num_classes
        x = x.reshape(-1, 196, 1, 1)
        x = x.softmax(dim=1)
        x = x.reshape(-1, self.num_classes, 14, 14)
        return x

class F_Confidence(nn.Module):
    """ Given the features map extracted from the first 3 layers of Resnet,
    the function needs to compute the Confidence Map: S. To make sure the
    output shape has the W and H constantly get the value of 14 given
    different value of input W and H, we use AdaptiveAveragePooling layer.

    The author noted that, the linear classifiers to compute the
    confidence map S are modeled as a convolution layer with C kernels
    of size 1 x 1. `S = W * X + b`, with * denoting convolution
    operation.

    Shape:
        - input x: (batch_size, 1024, H, W)
        - output: (batch_size, num_classes, 14, 14)

        Please note that the input image should be resize `width` or
        `height` to 224 px. Therefore, H or W in the above input x either
        will get the value of 14. For example, the valid input x should be:
        [batch_size, 1024, 14, 16] or [batch_size, 1024, 21, 14], shouldn't
        be: [batch_size, 1024, 21, 20].
    """
    def __init__(self, num_classes=10):
        super(F_Confidence, self).__init__()
        self.adap_average_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, bias=True)
        )
        self.conv.apply(lambda x: init_conv_gaussian(x, fill_bias=0))

    def forward(self, x):
        x = self.adap_average_pool(x)
        x = self.conv(x)
        return x

class F_LabelRegularization(nn.Module):
    """ Given the Weighted Attention Map: U, the label regularization
    function is required to estimate the label confidences based on label
    spatial information from U. (Refer to paper to understand what is the
    Weighted Attention Map U).

    The first 2 layers are 1 x 1 convolutions with 512 filters. The third
    layers is the special CNN named GroupConv2D using 2048 filters
    with 14 x 14 kernels. The filters of the third convolution layer are
    grouped, with each group of 4 kernels corresponding to one feature
    channel of the input feature map. The operation becomes equivalent
    to having 512 conv layers side by side, each seeing each 4 of the
    input channels, and producing 512 the output channels, and all will
    subsequently concatenated.

    Shape:
        - input x: (batch_size, 1024, H, W)
        - output: (batch_size, num_classes)
    """
    def __init__(self, num_classes=10):
        super(F_LabelRegularization, self).__init__()
        self.lr = nn.Sequential(
            nn.Conv2d(num_classes, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 2048, kernel_size=14, stride=1, groups=512, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=num_classes)
        )
        self.fc.apply(init_kaiming)
        self.lr.apply(init_conv_gaussian)

    def forward(self, x):
        x = self.lr(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
