import os

import torch
from PIL import Image
from torchvision.transforms import transforms

from face2vec import Face2Vec

face2vec = Face2Vec()

image_path = os.path.join(os.path.dirname(__file__), 'target.jpg')
image = Image.open(image_path)

faces = face2vec.mtcnn(image)
reconstructed = face2vec.reconstruct(image)

os.makedirs(os.path.join(os.path.dirname(__file__), 'results'), exist_ok=True)
cropped_face = faces.squeeze(0).cpu()
transforms.ToPILImage()(cropped_face).save(
    os.path.join(os.path.dirname(__file__), 'results', 'cropped.jpg'))
reconstructed_image = reconstructed.squeeze(0).cpu()
transforms.ToPILImage()(reconstructed_image).save(
    os.path.join(os.path.dirname(__file__), 'results', 'reconstructed.jpg'))
