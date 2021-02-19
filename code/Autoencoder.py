class Autoencoder(nn.Module):
    
    def __init__(self):
		super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=NUM_KERNELS[0], kernel_size=KERNEL_SIZES[0], stride=STRIDES[0], padding=0, padding_mode="zeros"),
                nn.BatchNorm2d(num_features=NUM_KERNELS[0]),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=NUM_KERNELS[0], out_channels=NUM_KERNELS[1], kernel_size=KERNEL_SIZES[1], stride=STRIDES[1], padding=0, padding_mode="zeros"),
                nn.BatchNorm2d(num_features=NUM_KERNELS[1]),
                nn.ReLU()
            )                            
        )

        self.decoder = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=NUM_KERNELS[1], out_channels=NUM_KERNELS[0], kernel_size=KERNEL_SIZES[1], stride=STRIDES[1], padding=0, padding_mode="zeros"),
                nn.BatchNorm2d(num_features=NUM_KERNELS[0]),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=NUM_KERNELS[0], out_channels=3, kernel_size=KERNEL_SIZES[0], stride=STRIDES[0], padding=0, padding_mode="zeros"),
                nn.BatchNorm2d(num_features=3),
                nn.Sigmoid()
            )
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
