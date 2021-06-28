import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from vae import VAELinear
from decoder import Decoder

vae = VAELinear()
vae.load_state_dict(torch.load("models/vae.h5"))
vae.eval()

def untargeted_generation():
    with torch.no_grad():
        for i in range(10):
            noise = torch.randn(1,2)
            generated_image = vae.decoder(noise)
            generated_image = generated_image[0].reshape(28, 28)
            plt.imshow(generated_image)
            plt.show()

def targeted_generation():
    model = vae.VAELinear()
    model.load_state_dict(torch.load("models/vae.h5"))
    model.eval()

    mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=10, shuffle=True)
    with torch.no_grad():
        for (img_original, _) in data_loader:
            img = img_original.reshape(-1, 28 * 28)
            decoded_image, mean, log_var = model(img)

            noise = torch.randn(1,2)
            generated_image = vae.decoder(noise)
            generated_image = generated_image[0].reshape(28, 28)
            plt.imshow(generated_image)
            plt.show()

untargeted_generation()