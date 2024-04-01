import torch
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

from models.vae import VariationalAutoEncoder
from models.mtcnn import MTCNN

EPOCHS = 10
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.PILToTensor()
])

train_dataset = CelebA('./data', transform=transform, split='train', download=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VariationalAutoEncoder(pretrained=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

mtcnn = MTCNN(image_size=128, device=device)

for epoch in range(EPOCHS):
    train_loss = 0

    print(f'Epoch {epoch + 1}/{EPOCHS}', '-' * 30)

    for batch_number, batch_data in enumerate(train_loader):
        images, _ = batch_data

        faces = mtcnn(images)

        optimizer.zero_grad()

        x, x_hat, mu, log_var = model(faces)
        loss, recon_loss, kld_loss = model.loss(x, x_hat, mu, log_var)
        loss.backward()

        train_loss += loss.item()
        optimizer.step()

        if batch_number % 100 == 0:
            print(f'\tBatch {batch_number}/{len(train_loader)}, Loss: {loss.item():.4f}')

    print('-' * 60)
    print(f'\tEpoch {epoch + 1}/{EPOCHS}, Loss: {train_loss / len(train_loader):.4f}')
    torch.save(model.state_dict(), f'weights/face2vec_epoch{epoch + 1}.pt')
