from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="yasirvision",
    version="0.1.0",
    author="Yasir Usman",
    description="A Python library simplifying OpenCV and MediaPipe into simple 20-line APIs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yasirusman85/yasirvision",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "opencv-python",
        "mediapipe==0.10.14",
        "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
