from setuptools import setup, find_packages

setup(
    name="ds_library",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers",
        "umap-learn",
        "matplotlib",
        "numpy",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "ds_library=ds_library.cli:main",
        ],
    },
    author="Abdellah WALID",
    author_email="abdellahwalid04@gmail.com",
    description="A data science library for text embedding, clustering, dimensionality reduction, and visualization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/abdwalid04/ds_library",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
