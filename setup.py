import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

SRC_REPO = "src"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    description="a small python package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
)
