# Copyright 2015 Ricequant All Rights Reserved
from setuptools import setup, find_packages


def readfile(filename):
    with open(filename, mode="rt") as f:
        return f.read()

setup(
    name='rqrisk',
    version='0.0.7',
    url="https://www.ricequant.com/",
    packages=find_packages(),
    author="Ricequant",
    install_requires=readfile("requirements.txt"),
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    description="risk calc"
)
