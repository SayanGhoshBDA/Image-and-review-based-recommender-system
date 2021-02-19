import torch
from torch import nn
import pytorch_ssim




class Custom_Loss_for_Autoencoder(nn.Module):
    def __init__(self, window_size=6):
        super(Custom_Loss_for_Autoencoder, self).__init__()
        self.ssim = pytorch_ssim.SSIM(window_size=window_size)
        self.mse = nn.MSELoss()
    
    def forward(self, reconstructed_images, images):
        l1 = self.mse(reconstructed_images, images)
        l2 = self.ssim(reconstructed_images, images)
        return l1 - l2
