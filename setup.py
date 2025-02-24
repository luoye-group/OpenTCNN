from gettext import install

import setuptools

from tcnn.version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()


package_name = "tcnn"
setuptools.setup(
    name=package_name,
    version=__version__,
    author="Mingbo Li@XMU",
    author_email="limingbo@stu.xmu.edu.cn",
    description="A implementation of the Tropical Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nittup/cTCNN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: apache 3.0",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
)
