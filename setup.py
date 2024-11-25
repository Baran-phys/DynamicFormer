from setuptools import setup, find_packages

setup(
    name="torch-dra",
    version="0.1.0",
    description="Dynamic Range Actovator (DRA), a learnable activation function",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Baran Hashemi",
    author_email="baran.hashemi@tum.de",
    url="https://github.com/Baran-phys/DynamicFormer",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",  # Specify compatible PyTorch version
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
