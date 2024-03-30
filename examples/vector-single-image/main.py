import torch
from torchvision import transforms
import os
from PIL import Image
from models.face2vec import Face2Vec
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Face2Vec().to(device)
model.load_state_dict(
    torch.load(os.path.join(os.path.dirname(__file__), '../../weights/vae.pt'), map_location=device))

target_image = Image.open(os.path.join(os.path.dirname(__file__), 'target.jpg'))
target_image = target_image.resize((128, 128))
target_image = transforms.ToTensor()(target_image).unsqueeze(0).to(device)

_, reconstructed_image, _, _ = model(target_image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(transforms.ToPILImage()(target_image.squeeze(0).cpu()))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Reconstructed Image')
plt.imshow(transforms.ToPILImage()(reconstructed_image.squeeze(0).cpu()))
plt.axis('off')

plt.show()


