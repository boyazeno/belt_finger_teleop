from setuptools import setup, find_packages

# python dependencies listed here will be automatically installed with the package
install_deps = []

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "belt_finger_control",
    version = "0.0.0",
    author = "Boya",
    author_email = "author.email",
    description = ("A package for control the belt finger and the gripper close/open."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license = "",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=install_deps,
    python_requires='>=3.6',
)
