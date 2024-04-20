import os

BASE_PATH = os.path.dirname(__file__)
MODULE_PATH = os.environ.get('FACE2VEC_MODULE_PATH') or os.environ.get('MODULE_PATH') or os.path.expanduser(
    '~/.face2vec/')
WEIGHTS_PATH = os.path.join(MODULE_PATH, 'weights')

pretrained_weights = {
    'onet': {
        'filename': 'onet.pt',
        'url': 'https://github.com/sturzamihai/face2vec/releases/download/v0.0.1-alpha/onet.pt',
        'md5': '4dbccd0886b13b1d17d08089ce02f1f4'
    },
    'pnet': {
        'filename': 'pnet.pt',
        'url': 'https://github.com/sturzamihai/face2vec/releases/download/v0.0.1-alpha/pnet.pt',
        'md5': 'e1de97352c413bfc5067f9940f6c9760'
    },
    'rnet': {
        'filename': 'rnet.pt',
        'url': 'https://github.com/sturzamihai/face2vec/releases/download/v0.0.1-alpha/rnet.pt',
        'md5': 'cf6209b4db20e0e1d586340333487767'
    },
    'vae': {
        'filename': 'vae.pt',
        'url': 'https://github.com/sturzamihai/face2vec/releases/download/v0.0.1-alpha/vae.pt',
        'md5': '4c3a9da8e701d4986c49ddaa188a3216'
    }
}
