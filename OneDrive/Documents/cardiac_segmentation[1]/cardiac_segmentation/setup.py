#!/usr/bin/env python3
"""
Setup script for Cardiac MRI Segmentation System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open("requirements.txt") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="cardiac-mri-segmentation",
    version="1.0.0",
    description="End-to-end cardiac MRI segmentation system using U-Net",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Cardiac AI Team",
    author_email="contact@example.com",
    url="https://github.com/example/cardiac-segmentation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cardiac-train=main:main",
            "cardiac-evaluate=evaluate:main",
            "cardiac-webapp=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    zip_safe=False,
    keywords="medical-imaging, deep-learning, segmentation, cardiac, mri, pytorch, u-net",
    project_urls={
        "Bug Reports": "https://github.com/example/cardiac-segmentation/issues",
        "Source": "https://github.com/example/cardiac-segmentation",
        "Documentation": "https://cardiac-segmentation.readthedocs.io/",
    },
)