# Copyright 2015 Ricequant All Rights Reserved
import versioneer
from setuptools import setup, find_packages


def readfile(filename):
    with open(filename, mode="rt") as f:
        return f.read()

setup(
    name='rqrisk',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url="https://www.ricequant.com/",
    packages=find_packages(),
    author="Ricequant",
    install_requires=readfile("requirements.txt"),
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    description="risk calc"
)
