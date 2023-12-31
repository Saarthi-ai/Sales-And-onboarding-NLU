from setuptools import setup, find_packages

setup(
    name='saarthi_train',
    version='2.0.0',
    packages=find_packages(),
    author='Sujay Rokade',
    author_email='sujay.rokade@saarthi.ai',
    description='Library for training AI models. Written in PyTorch.',
    long_description=open('README.md').read(),
    install_requires=[
        "aiohttp>=3.8.3",
        "aiosignal>=1.3.1",
        "anyio>=3.6.2",
        "async-timeout>=4.0.2",
        "attrs>=22.2.0",
        "charset-normalizer>=2.1.1",
        "click>=8.1.3",
        "fastapi>=0.89.1",
        "filelock>=3.9.0",
        "frozenlist>=1.3.3",
        "fsspec>=2023.1.0",
        "h11>=0.14.0",
        "httptools>=0.5.0",
        "huggingface-hub>=0.12.0",
        "idna>=3.4",
        "joblib>=1.2.0",
        "multidict>=6.0.3",
        "packaging>=23.0",
        "pandas>=1.5.3",
        "protobuf>=3.20.1",
        "pydantic>=1.10.4",
        "python-dateutil>=2.8.2",
        "python-dotenv>=0.21.1",
        "torch>=2.0.1",
        "lightning>=2.0.5",
        "lightning-utilities>=0.9.0",
        "pytz>=2022.7.1",
        "pyyaml>=6.0",
        "regex>=2022.10.31",
        "requests>=2.28.2",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "sentencepiece>=0.1.99",
        "sniffio>=1.3.0",
        "starlette>=0.22.0",
        "tensorboardx>=2.5.1",
        "threadpoolctl>=3.1.0",
        "tokenizers>=0.13.2",
        "torchmetrics>=0.11.0",
        "tqdm>=4.64.1",
        "transformers>=4.31.0",
        "triton>=2.0.0",
        "urllib3>=1.26.14",
        "uvicorn>=0.21.1",
        "uvloop>=0.17.0",
        "watchfiles>=0.18.1",
        "websockets>=10.4",
        "yarl>=1.8.2",
        "balanced-loss>=0.1.0",
        "optuna==3.3.0"
    ]
)   