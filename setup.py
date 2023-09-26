import io
import os
import re

from setuptools import find_packages, setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type("")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    name="opensr-test",
    version="0.0.5",
    url="https://github.com/ESAOpenSR/opensr-test",
    license="MIT",
    author="Cesar Aybar Camacho",
    author_email="csaybar@gmail.com",
    description="A comprehensive benchmark for real-world Sentinel-2 imagery super-resolution",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(
        exclude=("tests",), include=["opensr_test", "opensr_test.*"]
    ),
    install_requires=[
        "torch", "numpy", "matplotlib", "scikit-image", 
        "scikit-learn", "lpips", "requests", "kornia"
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
