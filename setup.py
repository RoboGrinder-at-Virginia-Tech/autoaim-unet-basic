from setuptools import setup, find_packages

setup(
    name="armor_unet",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1",
        "torchvision>=0.16",
        "pytorch-lightning>=2.2",
        "albumentations>=1.3",
        "numpy",
        "pillow",
        "matplotlib",
        "wandb>=0.16",
        "roboflow",
    ],
)