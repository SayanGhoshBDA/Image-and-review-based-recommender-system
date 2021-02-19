import torch
from torch import nn
from torch.nn import functional as F

from constants import *




class Attention(nn.Module):
    def __init__(self, query_dim, key_or_value_dim, units):
        super(Attention, self).__init__()
        self.units = units
        self.W1 = nn.Linear(in_features=query_dim, out_features=units)
        self.W2 = nn.Linear(in_features=key_or_value_dim, out_features=units)
        self.V = nn.Linear(in_features=units, out_features=1)
    
    def forward(self, query, key_or_value):
        query = torch.unsqueeze(query, 1)
        b, ch, h, w = key_or_value.size()
        key_or_value = key_or_value.view(b, ch, h*w).permute(0,2,1)
        intermediate_state = F.tanh(self.W1(query) + self.W2(key_or_value))
        attention_score = self.V(intermediate_state)
        attention_weights = F.softmax(attention_score, dim=1)
        weighted_values = key_or_value*attention_weights
        weighted_sum = torch.sum(weighted_values, axis=1)
        return weighted_sum
        
        
        
class Compound_Model(nn.Module):
    def __init__(self,num_users,num_items,user_embedding_dim,item_embedding_dim,attention_units,autoencoder,is_train):
        super(Compound_Model,self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.user_embedding_dim = user_embedding_dim
        self.item_embedding_dim = item_embedding_dim
        self.is_train = is_train

        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=user_embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=item_embedding_dim)

        self.encoder = autoencoder.encoder
        
        self.attention_1 = Attention(self.user_embedding_dim + self.item_embedding_dim, NUM_KERNELS[0], attention_units[0])
        self.attention_2 = Attention(NUM_DENSE_LAYER_UNITS[0] + NUM_KERNELS[0], NUM_KERNELS[1], attention_units[1])

        self.dense_layer_1 = nn.Sequential(nn.Linear(self.user_embedding_dim + self.item_embedding_dim, NUM_DENSE_LAYER_UNITS[0]),nn.BatchNorm1d(NUM_DENSE_LAYER_UNITS[0]))
        self.dense_layer_2 = nn.Sequential(nn.Linear(NUM_DENSE_LAYER_UNITS[0] + NUM_KERNELS[0], NUM_DENSE_LAYER_UNITS[1]),nn.BatchNorm1d(NUM_DENSE_LAYER_UNITS[1]))

        self.text_embedding_layer = nn.Sequential(nn.Dropout(0.2), nn.Linear(NUM_DENSE_LAYER_UNITS[1] + NUM_KERNELS[1], 768))

        self.output_layer = nn.Sequential(nn.Dropout(0.1), nn.Linear(NUM_DENSE_LAYER_UNITS[1] + NUM_KERNELS[1], 1))
    
    def forward(self, user_ids, item_ids, images):
        ue = self.user_embedding(user_ids)
        ie = self.item_embedding(item_ids)
        q = torch.cat((ue, ie), -1)

        kv = list(self.encoder.children())[0](images)
        c = self.attention_1(q, kv)
        c = c * torch.tensor(0.0)
        d = F.relu(self.dense_layer_1(q))
        q = torch.cat((d, c), -1)

        kv = list(self.encoder.children())[1](kv)
        c = self.attention_2(q, kv)
        c = c * torch.tensor(0.0)
        d = F.relu(self.dense_layer_2(q))
        q = torch.cat((d, c), -1)

        x = F.tanh(self.output_layer(q)).view(-1)
        if self.is_train:
            bert_out = F.tanh(self.text_embedding_layer(q))
            return bert_out,x 
        return x
