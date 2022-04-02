import os
from pathlib import Path

from setuptools import find_packages, setup


requirements = {
    "install": [
        "onnxruntime>=1.4.0",
        "librosa>=0.8.0",
        "sentencepiece>=0.1.91,!=0.1.92",
        "typeguard>=2.7.0",
        "PyYAML>=5.1.2",
        "g2p-en",
        "jamo==0.4.1",  # For kss
    ],
    "test": [
        "torch>=1.3.0",
        "espnet",
        "pytest"
    ]
}

setup(
    name="espnet_onnx",
    version="0.0.1",
    url="https://github.com/Masao-Someki/espnet_onnx",
    author="Masao Someki",
    author_email="masao.someki@gmail.com",
    description="ONNX Wrapper for ESPnet",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="espnet onnxruntime",
    license="MIT",
    packages=["espnet_onnx", "espnet_onnx.export"],
    # #448: "scripts" is inconvenient for developping because they are copied
    # scripts=get_all_scripts('espnet/bin'),
    install_requires=requirements["install"],
    tests_require=requirements["test"],
    python_requires=">=3.7.0",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)