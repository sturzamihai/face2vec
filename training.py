import torch
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

from models.face2vec import Face2Vec

EPOCHS = 10
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True),
    transforms.ToTensor()
])

train_dataset = CelebA('./data', transform=transform, split='train', download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Face2Vec().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    train_loss = 0
    for batch in train_loader:
        images, _ = batch
        images = images.to(device)

        optimizer.zero_grad()

        x, x_hat, mu, log_var = model(images)
        loss, recon_loss, kld_loss = model.loss(x, x_hat, mu, log_var)
        loss.backward()

        train_loss += loss.item()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss / len(train_loader):.4f}')
    torch.save(model.state_dict(), f'face2vec_epoch{epoch + 1}.pt')
