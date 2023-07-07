from setuptools import setup, find_packages
from setuptools.dist import Distribution
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()


setup(
    name="smart-tfrecord-writer",
    version="0.1",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/caharper/smart-tfrecord-writer.git",
    author="Clay Harper",
    author_email="caharper@smu.edu",
    license="Apache License 2.0",
    install_requires=[
        "tensorflow-datasets>=4.9.2",
        "tensorflow>=2.4.1",
        "numpy>=1.19.2",
        "scikit_learn>=1.3.0",
        "tqdm>=4.60.0",
    ],
    extras_require={},
    python_requires=">=3.7",
    distclass=Distribution,
    packages=find_packages(exclude=("*_test.py",)),
    include_package_data=True,
)
