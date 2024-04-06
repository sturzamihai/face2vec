import os

import torch
from PIL import Image
from torchvision.transforms import transforms

from models.mtcnn import MTCNN
from models.vae import VariationalAutoEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=128, device=device)
vae = VariationalAutoEncoder().to(device)
state_dict_path = os.path.join(os.path.dirname(__file__), '../../weights/vae_epoch3.pt')
vae.load_state_dict(torch.load(state_dict_path, map_location=device))

image_path = os.path.join(os.path.dirname(__file__), 'target.jpg')
image = Image.open(image_path)

faces = mtcnn(image)
x, x_hat, mu, log_var = vae(faces.unsqueeze(0).to(device))

os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
cropped_face = faces.squeeze(0).cpu()
transforms.ToPILImage()(cropped_face).save(
    os.path.join(os.path.dirname(__file__), 'results', 'cropped.jpg'))
reconstructed_image = x_hat.squeeze(0).cpu()
transforms.ToPILImage()(reconstructed_image).save(
    os.path.join(os.path.dirname(__file__), 'results', 'reconstructed.jpg'))
