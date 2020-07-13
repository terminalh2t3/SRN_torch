""" Dataloader """
import os
import math
import torch
import torch.utils.data
import pandas as pd
from PIL import Image
import numpy as np
from scipy import sparse
import h5py
from srn.util import convert_categorical, processing_fn

class SRNDataloader(torch.utils.data.Dataset):
    def __init__(self,
                 data_path="data/train.csv",
                 label_path="data/labels.txt",
                 images_path="data/images",
                 tag_delimiter="|",
                 mode="train"):
        self.tag_delimiter = tag_delimiter
        self.images_path = images_path
        self.mode = mode

        # Load all labels
        with open(label_path, 'r') as f:
            all_labels = f.read().split("\n")
        self.labels = all_labels

        # Load data
        df = pd.read_csv(data_path)
        df.columns = ["id", "tags"]
        df.tags = df.tags.str.split(self.tag_delimiter)
        df['target'] = df.tags.apply(lambda x: convert_categorical(self.labels, x))
        self.df = df

        # Calculate weight to prevent imbalance training data
        weights = torch.zeros([len(self.labels)])
        all_targets = np.array(list(df.target.values))
        for i in range(len(self.labels)):
            t = all_targets[:, i]
            pos = (t == 1).sum()
            neg = (t == 0).sum()
            weights[i] = neg / (pos + 1e-6)
        self.weights = weights

    def get_num_classes(self):
        return len(self.labels)

    def get_data_pos_weights(self):
        return self.weights

    def load_img(self, image_id):
        image_filename = "{}.jpg".format(image_id)
        image_path = os.path.join(self.images_path, image_filename)
        img = Image.open(image_path).convert("RGB")
        img = processing_fn(self.mode)(img)
        return img

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        target = torch.from_numpy(self.df.target.iloc[idx]).float()
        img_id = self.df.id.iloc[idx]
        img = self.load_img(img_id)
        return img, target

class CustomDataset5k(torch.utils.data.DataLoader):
    def __init__(self,
                 data_path="data/custom",
                 images_path="/home/lab/custom_data/tag_suggestion/classes10k/images",
                 max_num_training=None,
                 dataset_name="5k",
                 mode="train"):
        self.data_path = os.path.join(data_path, dataset_name)
        self.label_path = os.path.join(self.data_path, "classes{}.csv".format(dataset_name))
        self.h5_path = os.path.join(self.data_path, mode, "data.h5")
        self.weights_path = os.path.join(self.data_path, "pos_weights.txt")
        self.images_path = images_path
        self.mode = mode

        # Load labels
        self.df_label = pd.read_csv(self.label_path)

        # Load h5 data
        item_ids, targets = self.h5_to_sparse(self.h5_path)
        self.item_ids = item_ids
        self.targets = targets

        # Load weights
        weights = np.loadtxt(self.weights_path)
        self.weights = torch.from_numpy(weights)

        # Set maximum samples to training or testing
        if max_num_training is None:
            self.max_num_training = self.item_ids.shape[0]
        else:
            self.max_num_training = max_num_training

    @staticmethod
    def h5_to_sparse(h5_path, chunksize=10000):
        h5 = h5py.File(h5_path, "r")
        start_idx = 0

        # All item_ids in h5
        item_ids = h5["item_ids"][:]
        num_items = item_ids.shape[0]

        list_data = []
        # Convert all target data to scipy
        for _ in range(math.ceil(num_items / chunksize)):
            to_idx = start_idx + chunksize
            chk_targets = h5["targets"][start_idx:to_idx]
            csr_targets = sparse.csr_matrix(chk_targets)
            list_data.append(csr_targets)

            start_idx = to_idx
        # Stack list
        targets = sparse.vstack(list_data)
        return item_ids, targets

    def get_num_classes(self):
        return self.df_label.shape[0]

    def get_data_pos_weights(self):
        return self.weights

    def load_img(self, image_id):
        image_filename = "{}.jpg".format(image_id)
        image_path = os.path.join(self.images_path, image_filename)
        img = Image.open(image_path).convert("RGB")
        img = processing_fn(self.mode)(img)
        return img

    def download_images(self, num_worker=1):
        pass

    def __len__(self):
        return self.max_num_training

    def __getitem__(self, idx):
        # return torch.ones(3, 224, 224), torch.ones(5000,)
        try:
            target = torch.from_numpy(self.targets[idx].toarray().flatten()).float()
            img_id = self.item_ids[idx]
            img = self.load_img(img_id)
        except Exception as ex:
            print("[Dataloader] Something wrong while load index {}: {}".format(idx, ex))
            rand_idx = np.random.randint(self.__len__())
            return self.__getitem__(rand_idx)
        return img, target
