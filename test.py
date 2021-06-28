import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import vae


mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=64, shuffle=True)

model = vae.VAELinear()
model.load_state_dict(torch.load("models/vae.h5"))
model.eval()

count = 0

fig=plt.figure()
SAMPLE = 10
for epoch in range(1):
    for (img_originals, _) in data_loader:
        img_original = img_originals[0]
        img = img_original.reshape(-1, 28*28)
        recon,_,_ = model(img)
        recon_original = recon.reshape(28,28)
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