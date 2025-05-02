from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='scEpiSparX',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    author='Cliff Meyer',
    description='A package for single-cell epigenomics analysis',
    url='https://github.com/yourusername/scEpiSparX',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
)

