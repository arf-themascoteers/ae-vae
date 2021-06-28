import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from vae import VAELinear

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
    mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=10, shuffle=True)
    fig = plt.figure()
    SAMPLE = 10
    count = 0
    with torch.no_grad():
        for (img_originals, _) in data_loader:
            img_original = img_originals[0]
            img = img_original.reshape(-1, 28 * 28)
            recon, _, _ = vae.encoder(img)

            NR = 0.2
            noise = torch.randn(1,2)
            decoded_image = recon*(1-NR) + noise*NR
            generated_image = vae.decoder(decoded_image)

            recon_original = generated_image.reshape(28, 28)
            original = img_original[0].detach().numpy()
            made = recon_original.detach().numpy()
            count = count + 1
            fig.add_subplot(SAMPLE, 2, count)
            plt.imshow(original)
            count = count + 1
            fig.add_subplot(SAMPLE, 2, count)
            plt.imshow(made)
            if count >= SAMPLE * 2:
                break
        plt.show()


targeted_generation()