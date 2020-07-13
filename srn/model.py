""" Models """
import torch
import torch.nn as nn
from srn.layer import F_CNN, F_Classification, F_Attention
from srn.layer import F_Confidence, F_LabelRegularization

class MainNet(nn.Module):
    """ MainNet class follows the structure of ResNet152

    Shape:
        - input image X: (batch_size, 3, H, W)
        - output: (batch_size, num_classes)
    """
    def __init__(self, num_classes=10):
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        self.f_cnn = F_CNN()
        self.f_cls = F_Classification(num_classes)

    def forward(self, x):
        x = self.f_cnn(x)
        x = self.f_cls(x)
        # x = torch.sigmoid(x)
        return x

class SpatialRegularizationNet(nn.Module):
    """
    An implementation of independent Spatial Regularization Network
    that can take a batch of images as the input. The model will
    return the confidence `yhat_att` that allow to train this model
    independently to the mainnet. See that dashed lines in the figure
    2 in the paper for futher information.

    Shape:
        - input image X: (batch_size, 3, H, W)
        - output: (batch_size, num_classes)
    """
    def __init__(self, mainnet):
        super(SpatialRegularizationNet, self).__init__()
        self.mainnet = mainnet
        # freeze mainnet model
        for _, param in self.mainnet.named_parameters():
            param.requires_grad = False
        num_classes = self.mainnet.num_classes

        self.att = F_Attention(num_classes=num_classes)
        self.cnd = F_Confidence(num_classes=num_classes)

    def forward(self, x):
        feature_map = self.mainnet.f_cnn(x)

        attention_map = self.att(feature_map)                   # Attention Map: A (b, 14, 14, C)
        confidence_map = self.cnd(feature_map)                  # Confidence Map: S
        x = attention_map * confidence_map                      # element-wise multiply
        x = torch.sum(x.view(x.size(0), x.size(1), -1), dim=2)  # sum pooling

        return x

class FuseNet(nn.Module):
    """
    FuseNet included MainNet and Spatial Regularization Net.
    The final label confidensces are aggregation of the outputs
    of main net and SRN, `y = alpha * y_cls + (1 - alpha) * y_sr`
    where `alpha` is weighting factor.

    Mode:
        - sr: mode train `label regularization` f_sr by freezing
        all other parameter of SRN.
        - finetune: mode finetune the whole network including
        mainnet and srn
    """
    def __init__(self, mainnet, srn, mode, alpha=0.5):
        super(FuseNet, self).__init__()
        if mode not in ["sr", "finetune", "infer"]:
            raise "The mode should be one of 'sr' or 'finetune' or 'infer'."

        self.alpha = alpha
        self.mainnet = mainnet
        self.srn = srn

        self.mode = mode
        num_classes = self.mainnet.num_classes
        self.lr = F_LabelRegularization(num_classes=num_classes)

        if self.mode == "finetune":
            print("Mode: FUSE. Unfreezed mainnet, att, cnd layers.")
            for _, param in self.mainnet.f_cls.named_parameters():
                param.requires_grad = True
            for _, param in self.srn.att.named_parameters():
                param.requires_grad = True
            for _, param in self.srn.cnd.named_parameters():
                param.requires_grad = True
        elif self.mode == "sr":
            print("Mode: SR. Freezed mainnet, att, cnd layers.")
            for _, param in self.mainnet.f_cls.named_parameters():
                param.requires_grad = False
            for _, param in self.srn.att.named_parameters():
                param.requires_grad = False
            for _, param in self.srn.cnd.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        feature_map = self.mainnet.f_cnn(x)

        # Compute y_sr from Spatial Regularization Net
        attention_map = self.srn.att(feature_map)               # Attention Map: A
        confidence_map = self.srn.cnd(feature_map)              # Confidence Map: S

        x_sr = attention_map * torch.sigmoid(confidence_map)      # element-wise multiply
        y_sr = self.lr(x_sr)                                      # label regularize

        if self.mode == "sr":
            # in case of mode is `sr`:
            return y_sr
        elif self.mode == "infer":
            # Compute y_cls from MainNet
            res_cls = self.mainnet.f_cls(feature_map)
            y_cls = res_cls

            # Element wise addition
            y_hat = self.alpha * y_sr + (1 - self.alpha) * y_cls
            return y_hat
        elif self.mode == "finetune":
            # Compute y_cls from MainNet
            res_cls = self.mainnet.f_cls(feature_map)
            y_cls = res_cls

            # Element wise addition
            y_hat = self.alpha * y_sr + (1 - self.alpha) * y_cls

            x_att = attention_map * confidence_map
            y_att = torch.sum(x_att.view(x_att.size(0), x_att.size(1), -1), dim=2)

            return y_hat, y_att
