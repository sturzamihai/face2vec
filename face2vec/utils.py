import hashlib
import os

import requests

from face2vec.config import WEIGHTS_PATH, pretrained_weights


def calculate_md5_hash(file_path):
    with open(file_path, 'rb') as f:
        md5 = hashlib.md5()
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)

    return md5.hexdigest()


def is_model_downloaded(filename, md5_hash):
    file_path = os.path.join(WEIGHTS_PATH, filename)

    if not os.path.exists(file_path):
        return False

    calculated_md5_hash = calculate_md5_hash(file_path)

    return calculated_md5_hash == md5_hash


def get_model_weights(model_name):
    if model_name not in pretrained_weights.keys():
        raise ValueError(f'Model {model_name} not found in pretrained_weights.')

    if not is_model_downloaded(filename=pretrained_weights[model_name]["filename"],
                               md5_hash=pretrained_weights[model_name]["md5"]):
        download_model(download_url=pretrained_weights[model_name]["url"],
                       filename=pretrained_weights[model_name]["filename"],
                       md5_hash=pretrained_weights[model_name]["md5"])

    return os.path.join(WEIGHTS_PATH, pretrained_weights[model_name]["filename"])


def download_model(download_url, filename, md5_hash, verbose=True):
    if verbose:
        print(f'Downloading model from {download_url}...')

    response = requests.get(download_url)
    file_path = os.path.join(WEIGHTS_PATH, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as f:
        f.write(response.content)

    if verbose:
        print(f'Downloaded model to {file_path}.')
        print(f'Calculating MD5 hash...')

    calculated_md5_hash = calculate_md5_hash(file_path)

    if calculated_md5_hash != md5_hash:
        raise ValueError(f'MD5 hash mismatch. Expected {md5_hash}, got {calculated_md5_hash}.')

    if verbose:
        print(f'MD5 hash matches. Model downloaded successfully.')

    return file_path
