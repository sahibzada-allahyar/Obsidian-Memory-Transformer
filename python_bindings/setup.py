from setuptools import setup, find_packages
from setuptools.dist import Distribution
import os
import sys

class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package"""
    def has_ext_modules(self):
        return True

def get_version():
    """Get version from version.h"""
    with open(os.path.join(os.path.dirname(__file__), '..', 'version.h'), 'r') as f:
        for line in f:
            if line.startswith('#define LTM_VERSION'):
                return line.split('"')[1]
    return '0.1.0'  # Default version if not found

# Check Python version
if sys.version_info < (3, 7):
    sys.exit('Python >= 3.7 is required')

# Get long description from README
with open(os.path.join(os.path.dirname(__file__), '..', 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ltm-transformer',
    version=get_version(),
    author='Sahibzada Allahyar',
    author_email='allahyar@singularityresearch.org',
    description='Long-term Memory Transformer with Titan-inspired architecture',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/singularityresearch/ltm-transformer',
    packages=find_packages(),
    package_data={
        'ltm': ['*.so', '*.pyd', 'config/*.yaml'],
    },
    distclass=BinaryDistribution,
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.0',
        'torch>=1.9.0',
        'pyyaml>=5.1',
        'tqdm>=4.45.0',
        'tensorboard>=2.4.0',
        'transformers>=4.5.0',
        'datasets>=1.6.0',
        'sentencepiece>=0.1.96',
        'tokenizers>=0.10.3',
        'wandb>=0.12.0',
        'pytest>=6.0.0',
        'pytest-benchmark>=3.4.0',
    ],
    extras_require={
        'dev': [
            'black',
            'isort',
            'flake8',
            'mypy',
            'pytest-cov',
            'sphinx',
            'sphinx-rtd-theme',
        ],
        'distributed': [
            'mpi4py>=3.0.0',
            'horovod>=0.21.0',
        ],
        'quantization': [
            'onnx>=1.9.0',
            'onnxruntime>=1.8.0',
        ],
        'serving': [
            'fastapi>=0.65.0',
            'uvicorn>=0.14.0',
            'grpcio>=1.38.0',
            'grpcio-tools>=1.38.0',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    entry_points={
        'console_scripts': [
            'ltm-train=ltm.trainer:main',
            'ltm-infer=ltm.inference:main',
            'ltm-serve=ltm.server:main',
            'ltm-convert=ltm.tools.convert:main',
            'ltm-benchmark=ltm.tools.benchmark:main',
        ],
    },
    project_urls={
        'Documentation': 'https://ltm-transformer.readthedocs.io/',
        'Source': 'https://github.com/singularityresearch/ltm-transformer',
        'Tracker': 'https://github.com/singularityresearch/ltm-transformer/issues',
    },
)
