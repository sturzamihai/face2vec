import torch
from torchvision import transforms
import os
from PIL import Image
from models.mtcnn import MTCNN
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MTCNN(device=device)

target_image = Image.open(os.path.join(os.path.dirname(__file__), 'target.jpg'))
tensor_image = transforms.PILToTensor()(target_image).to(device)


pil_faces = model([target_image, target_image], save_path=os.path.join(os.path.dirname(__file__), 'pil_face.png'))
tensor_faces = model(tensor_image, save_path=os.path.join(os.path.dirname(__file__), 'tensor_face.png'))

print("PIL:", pil_faces)
print("Tensor:", tensor_faces)

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(transforms.ToPILImage()(target_image.squeeze(0).cpu()))
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.title('Reconstructed Image')
# plt.imshow(transforms.ToPILImage()(reconstructed_image.squeeze(0).cpu()))
# plt.axis('off')
#
# plt.show()


