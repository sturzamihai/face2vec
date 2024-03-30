import torch
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

from models.vae import VariationalAutoEncoder
from models.mtcnn import MTCNN

EPOCHS = 10
BATCH_SIZE = 32

train_dataset = CelebA('./data', split='train', download=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VariationalAutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

mtcnn = MTCNN(device=device)

for epoch in range(EPOCHS):
    train_loss = 0

    print(f'Epoch {epoch + 1}/{EPOCHS}', '-' * 10)

    for batch_number, batch_data in enumerate(train_loader):
        images, _ = batch_data
        images = images.to(device)

        faces = mtcnn(images)

        if faces is None:
            continue

        optimizer.zero_grad()

        batch_loss = 0
        for face in faces:
            face = face.unsqueeze(0)
            x, x_hat, mu, log_var = model(face)
            loss, recon_loss, kld_loss = model.loss(x, x_hat, mu, log_var)
            loss.backward()

            batch_loss += loss.item()
            optimizer.step()

        train_loss += batch_loss / len(faces)

        if batch_number % 100 == 0:
            print(f'\tBatch {batch_number}/{len(train_loader)}, Loss: {batch_loss / len(faces):.4f}')

    print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss / len(train_loader):.4f}')
    print('-' * 20)
    torch.save(model.state_dict(), f'weights/face2vec_epoch{epoch + 1}.pt')
