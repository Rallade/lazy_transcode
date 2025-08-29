from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Lazy Transcode - A smart video transcoding utility with VMAF-based quality optimization"

setup(
    name="lazy-transcode",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A smart video transcoding utility with VMAF-based quality optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lazy-transcode",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video :: Conversion",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tqdm>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
        "full": [
            "psutil>=5.0.0",  # For CPU monitoring
        ],
    },
    entry_points={
        "console_scripts": [
            "lazy-transcode=lazy_transcode.cli:main_transcode",
            "lazy-transcode-manager=lazy_transcode.cli:main_manager",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
