from PIL import Image
import numpy as np

from face2vec import Face2Vec

face2vec = Face2Vec()

t1 = Image.open('targets/t1.jpg').convert("RGB")
t2 = Image.open('targets/t2.jpg').convert("RGB")
b1 = Image.open('targets/t3.png').convert("RGB")

v1 = face2vec(t1).detach().numpy()[0]
v2 = face2vec(t2).detach().numpy()[0]
bv1 = face2vec(b1).detach().numpy()[0]

# Calculate the cosine similarity between the vectors
amplification = 5
v1_v2 = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) * amplification
v1_bv1 = np.dot(v1, bv1) / (np.linalg.norm(v1) * np.linalg.norm(bv1)) * amplification
v2_bv1 = np.dot(v2, bv1) / (np.linalg.norm(v2) * np.linalg.norm(bv1)) * amplification

print(f"Similarity between t1 and t2: {v1_v2}")
print(f"Similarity between t1 and b1: {v1_bv1}")
print(f"Similarity between t2 and b1: {v2_bv1}")

biggest = max(v1_v2, v1_bv1, v2_bv1)
print(f"Biggest similarity: {biggest} (between {['t1 and t2', 't1 and b1', 't2 and b1'][[v1_v2, v1_bv1, v2_bv1].index(biggest)]})")


