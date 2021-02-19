import pandas as pd
import numpy as np
import os
import glob
from skimage import io
import PIL
from PIL import Image
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import pytorch_ssim

from transformers import AlbertTokenizer, AlbertForSequenceClassification

from constants import *
from Autoencoder import *
from Custom_Loss_for_Autoencoder import *
from Custom_Loss import *
from Custom_Dataloader import *
from Compound_Model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




auto = Autoencoder().to(device)
model = Compound_Model(NUM_USERS, NUM_ITEMS, USER_EMBEDDING_DIM, ITEM_EMBEDDING_DIM, attention_units=ATTENTION_UNITS, autoencoder=auto, is_train=True).to(device)

auto_criterion = Custom_Loss_for_Autoencoder()
criterion = Custom_Loss(0.5)

text_encoder = AlbertForPreTraining.from_pretrained("albert-base-v2").to(device)
for param in text_encoder.parameters():
    param.requires_grad = False

train_dataset = CustomDataset(TRAIN_DATA_FILE, PRODUCT_IMAGE_DIR, is_autoencoder=True, is_train=True)
val_dataset = CustomDataset(TEST_DATA_FILE, PRODUCT_IMAGE_DIR, is_autoencoder=True, is_train=False)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
num_batches = len(train_dataloader)


for epoch in np.arange(NUM_EPOCHS):
    t = time.time()
    auto.encoder = model.encoder
    auto.train()
    train_dataset.is_autoencoder = True

    for i, images in enumerate(train_dataloader):
        images = images.to(device)

        reconstructed_images = auto(images)
        auto_loss = auto_criterion(reconstructed_images, images)

        auto_loss.backward()
        auto_optimizer = torch.optim.Adam(auto.parameters(),lr=auto_LEARNING_RATE)
        auto_optimizer.step()
        auto_optimizer.zero_grad()

        left = (num_batches - i - 1) * (time.time() - t)
        t = time.time()
        print(f"\rEpoch: {epoch+1}/{NUM_EPOCHS},   Phase:1,   Step: {i+1}/{num_batches},   Time Left: {int(left//60)} m {int(left%60)} s,   Auto Loss: {auto_loss.item():.4f}", end="")

    model.encoder = auto.encoder
    model.is_train=True
    model.train()
    train_dataset.is_autoencoder = False
    print("")
    for i, (user_ids, item_ids, images, integer_sequence, attention_mask, ratings) in enumerate(train_dataloader):
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        images = images.to(device)
        integer_sequence = integer_sequence.to(device)
        attention_mask = attention_mask.to(device)
        ratings = ratings.to(device)
        bert_out = F.tanh(text_encoder(integer_sequence, attention_mask, output_hidden_states=True).hidden_states[0][:,0,:])
        bert_out = bert_out.to(device)

        predicted_text_embedding, predicted_rating = model(user_ids, item_ids, images)
        rating_loss = nn.MSELoss()(predicted_rating, ratings)
        combined_loss = criterion(predicted_rating, ratings, predicted_text_embedding, bert_out)
        combined_loss.backward()
        optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
        optimizer.step()
        optimizer.zero_grad()

        left = (num_batches - i - 1) * (time.time() - t)
        t = time.time()
        print(f"\rEpoch: {epoch+1}/{NUM_EPOCHS},   Phase:2,   Step: {i+1}/{num_batches},   Time Left: {int(left//60)} m {int(left%60)} s,   rating Loss: {rating_loss.item():.4f},   combined loss: {combined_loss.item():.4f}", end="")
    
    model.eval()
    model.is_train = False
    auto_val_loss = 0.0
    rating_val_loss = 0.0
    combined_val_loss = 0.0
    val_dataset.is_autoencoder = False
    with torch.no_grad():
        for user_ids, item_ids, images, ratings in val_dataloader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            images = images.to(device)
            ratings = ratings.to(device)
            predicted_rating = model(user_ids, item_ids, images)
            rating_val_loss += nn.MSELoss()(predicted_rating, ratings).item()
    
    rating_val_loss = rating_val_loss/float(len(val_dataloader))
    print(f"\nEpoch: {epoch+1}/{NUM_EPOCHS},   rating Val Loss: {rating_val_loss}")
    print("")
    PATH = './model.pth'
    torch.save(model.state_dict(),PATH)
    PATH = './autoencoder.pth'
    torch.save(auto.state_dict(),PATH)

print("Finished training")
