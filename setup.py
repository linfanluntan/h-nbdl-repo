from setuptools import setup, find_packages

setup(
    name="h-nbdl",
    version="0.1.0",
    description="Hierarchical Nonparametric Bayesian Dictionary Learning",
    author="[Author A]",
    author_email="author@university.edu",
    url="https://github.com/[user]/h-nbdl",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "pyyaml>=6.0",
        "tqdm>=4.65",
    ],
    extras_require={
        "hbm": ["pymc>=5.0", "arviz>=0.15"],
        "rl": ["gymnasium>=0.28", "stable-baselines3>=2.0"],
        "dev": ["pytest>=7.0", "black", "isort", "flake8"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
