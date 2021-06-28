import torch
from torchvision import datasets, transforms
import vae
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
import time

def calculate_loss(x, decoded, mean, log_var):
    reproduction_loss = F.binary_cross_entropy(decoded, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                              batch_size=64,
                                              shuffle=True)

    model = vae.VAELinear().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    writer = SummaryWriter()
    num_epochs = 5
    for epoch in range(num_epochs):
        first = True
        for (img_original, _) in data_loader:
            img_original = img_original.to(device)
            img = img_original.reshape(-1, 28*28)
            decoded_image, mean, log_var = model(img)
            loss = calculate_loss(img, decoded_image, mean, log_var)

            if first:
                writer.add_scalar("Loss/train", loss, epoch)
                sample_generated_images = decoded_image.reshape(-1,1,28, 28)
                grid = torchvision.utils.make_grid(img_original[0:5])
                writer.add_image(f"{epoch} - Original", grid)
                grid = torchvision.utils.make_grid(sample_generated_images[0:5])
                writer.add_image(f"{epoch} - Generated", grid)
                writer.flush()
                first = False

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"End of epoch {epoch}")

    torch.save(model.state_dict(), 'models/vae.h5')
    writer.close()

start = time.time()
train()
end = time.time()

print(f"Required time {end-start}")