""" utility """
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from scipy.ndimage import gaussian_filter

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

def init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

def init_conv_gaussian(m, std=0.005, fill_bias=None):
    if isinstance(m, nn.Conv2d):
        layer_weight = m.weight.detach().cpu().numpy()
        generated_filters = gaussian_filter(layer_weight, std)
        m.weight.data.copy_(torch.from_numpy(generated_filters))

        if fill_bias is not None:
            m.bias.data.fill_(fill_bias)

def convert_categorical(all_labels, list_labels):
    """
    Convert a list of label to one hot vectors
    Input:
        - all_labels: list of all labels
        - list_labels: list of labels that need to be converted
    """
    vector_len = len(all_labels)

    data = np.zeros(vector_len)
    idx = [all_labels.index(n) for n in list_labels]
    data[idx] = 1
    return data

def processing_fn(mode):
    normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )

    if mode == "train":
        preprocessing = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
    else:
        preprocessing = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
            ])

    return preprocessing
