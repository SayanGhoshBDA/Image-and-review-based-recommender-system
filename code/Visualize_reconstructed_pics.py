import numpy as np
import matplotlib.pyplot as plt
import torch

from Custom_Dataset import *
from Autoencoder import *
from constants import *




def visualize_reconstruction(original_images,noisy_images,reconstructed_images):
    if noisy_images is not None:
        figure,ax = plt.subplots(3,10,figsize=(20,6))
        pointers = {"Original":original_images,"Noisy":noisy_images,"Reconstructed":reconstructed_images}
    else:
        figure,ax = plt.subplots(2,10,figsize=(20,4))
        pointers = {"Original":original_images,"Reconstructed":reconstructed_images}


    indices = np.random.choice(range(original_images.shape[0]),10)
    for i in range(10):
        for j, image_group in enumerate(list(pointers.keys())):
            ax[j,i].imshow(pointers[image_group][indices[i]].swapaxes(0,2).swapaxes(0,1))
            ax[j,i].get_xaxis().set_visible(False)
            ax[j,i].set_yticks([])
            
            if i==0:
                ax[j,i].set_ylabel(image_group)
                
               
if __name__=="__main__":
	dataset = CustomDataset(TEST_DATA_FILE, PRODUCT_IMAGE_DIR, is_autoencoder=True, is_train=False)
	check_point = torch.load("autoencoder.pth", map_location="cpu")
	auto = Autoencoder()
	auto.load_state_dict(check_point)
	original_images = np.array([val_dataset.__getitem__(idx).numpy() for idx in np.random.choice(np.arange(len(dataset)), 100)])
	with torch.no_grad():
    	reconstructed_images = auto(torch.tensor(original_images)).cpu().numpy()
	visualize_reconstruction(original_images=original_images, noisy_images=None, reconstructed_images=reconstructed_images)

