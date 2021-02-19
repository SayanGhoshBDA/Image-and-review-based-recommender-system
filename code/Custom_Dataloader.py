import pandas as pd
import numpy as np
import os
import glob
from skimage import io
import PIL
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms

from transformers import AlbertTokenizer

from constants import *




class CustomDataset(Dataset):
    def __init__(self,transaction_file_path, image_dir_path, is_autoencoder, is_train):
        self.transaction_data = pd.read_csv(transaction_file_path, delimiter="\t", names=["user_id","item_id","rating","review"])
        self.image_dir_path = image_dir_path
        self.item_id__image_count__map = {item_id:len(glob.glob1(f"{self.image_dir_path}/P{item_id:08d}/",f"P{item_id:08d}*.png")) for item_id in self.transaction_data["item_id"]}
        self.is_autoencoder = is_autoencoder
        self.is_train = is_train
        self.tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
        self.hacker_string = " ".join(["a"]*(NUM_WORDS+2))
    
    def parse_image(self, filename):
        image = io.imread(filename)
        if len(image.shape)==2:
            image = np.stack((image,)*3,axis=-1)
        image = torch.tensor(image/255., dtype=torch.float32).permute(2,0,1)
        image = transforms.Resize((40,40))(image)
        return image
    
    def __len__(self):
        return len(self.transaction_data.index)
    
    def __getitem__(self, idx):
        user_id, item_id, rating, review = self.transaction_data.iloc[idx]
        filename = f"{self.image_dir_path}/P{item_id:08d}/P{item_id:08d}_{np.random.randint(0,self.item_id__image_count__map[item_id]):02d}.png"
        image = self.parse_image(filename)
        if not self.is_train:
            return user_id, item_id, image, torch.tensor((rating - 3.0)/2.0, dtype=torch.float32)
        if not self.is_autoencoder:
            tokens_info=self.tokenizer([str(review),self.hacker_string],padding=True,max_length=NUM_WORDS+2,truncation=True)
            integer_sequence = tokens_info["input_ids"][0]
            attention_mask = tokens_info["attention_mask"][0]
            return user_id, item_id, image, torch.tensor(integer_sequence), torch.tensor(attention_mask), torch.tensor((rating - 3.0)/2.0, dtype=torch.float32)
        return image

    
    
