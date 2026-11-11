from setuptools import find_packages, setup

requirements = {
    "install": [
        "setuptools>=38.5.1",
        "librosa>=0.10.2",
        "onnxruntime",
        "sentencepiece>=0.2.1",
        "typeguard==2.13.0",
        "PyYAML>=5.1.2",
        "g2p-en",
        "jamo==0.4.1",  # For kss
        "six",
        "numpy>=2.0.0",
        "espnet_tts_frontend",  # required for TTS preprocess.
    ],
    "test": ["torch>=2.11.0", "espnet>=202611", "onnx>=1.22.0", "onnxscript", "pytest"],
}

setup(
    name="espnet_onnx",
    version="0.2.2",
    url="https://github.com/Masao-Someki/espnet_onnx",
    author="Masao Someki",
    author_email="masao.someki@gmail.com",
    description="ONNX Wrapper for ESPnet",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="espnet onnxruntime",
    license="MIT",
    packages=find_packages(include=["espnet_onnx*"]),
    package_data={"espnet_onnx.export": ["convert_map.yml"]},
    install_requires=requirements["install"],
    tests_require=requirements["test"],
    python_requires=">=3.7.0",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
