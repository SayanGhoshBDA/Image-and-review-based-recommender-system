import torch
from torch import nn




class Custom_Loss(nn.Module):
    def __init__(self, lambda_):
        super(Custom_Loss, self).__init__()
        self.lambda_ = lambda_
        self.mse1 = nn.MSELoss()
        self.mse2 = nn.MSELoss()
    
    def forward(self, predicted_ratings, actual_ratings, predicted_text_embedding, actual_text_embedding):
        l1 = self.mse1(predicted_ratings, actual_ratings)
        l2 = self.mse2(predicted_text_embedding, actual_text_embedding)
        return self.lambda_*l1 + (1 - self.lambda_)*l2

    
