import os

from PIL import Image

from face2vec.face2vec import Face2Vec

face2vec = Face2Vec()

image_path = os.path.join(os.path.dirname(__file__), 'target.jpg')
image = Image.open(image_path)

embedding = face2vec(image)
print(embedding, embedding.shape)
