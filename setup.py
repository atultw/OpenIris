"""
Setup script for python_openiris package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "python_openiris" / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="python_openiris",
    version="1.0.0",
    description="Python implementation of OpenIris eye tracking with OpenCV",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="OpenIris Contributors",
    author_email="",
    url="https://github.com/ocular-motor-lab/OpenIris",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="eye-tracking pupil iris opencv computer-vision",
    entry_points={
        "console_scripts": [
            "openiris-demo=sample_eye_tracker:main",
        ],
    },
)
