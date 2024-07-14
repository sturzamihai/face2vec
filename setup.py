from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()

def requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
    name='face2vec',
    packages=['face2vec'],
    version='0.0.3-alpha',
    license='MIT',
    author='Mihai-George Sturza',
    author_email='contact@mihaisturza.ro',
    description='A Python library for transforming faces into feature vectors.',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/sturzamihai/face2vec',
    download_url='https://github.com/sturzamihai/face2vec',
    install_requires=requirements(),
)
