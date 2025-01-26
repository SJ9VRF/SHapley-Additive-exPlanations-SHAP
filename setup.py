from setuptools import setup, find_packages

setup(
    name="shap-implementation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.2",
        "scikit-learn>=0.24.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Pure Python implementation of SHAP",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/shap-implementation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
