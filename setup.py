#!/usr/bin/env python3

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="whisper-cli-enhanced",
    version="1.0.0",
    author="Patrick",
    description="Enhanced Whisper CLI with advanced user experience features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        req
        for req in requirements
        if not req.startswith("pystray") and not req.startswith("Pillow")
    ],
    extras_require={
        "tray": ["pystray>=0.19.4", "Pillow>=9.0.0"],
        "dev": ["pytest>=6.0", "pytest-cov>=2.0"],
    },
    entry_points={
        "console_scripts": [
            "whisper-cli=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
