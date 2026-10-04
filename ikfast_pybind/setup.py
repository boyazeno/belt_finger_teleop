from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os

# python dependencies listed here will be automatically installed with the package
install_deps = []

with open("README.md", "r") as fh:
    long_description = fh.read()

# class CustomInstallCommand(install):
#     def run(self):
#         # Run the base class install method
#         install.run(self)

#         # Path to your CMakeLists.txt file
#         current_directory = os.path.abspath(os.path.dirname(__file__))
#         print("!"*80)
#         print(current_directory)
#         os.chdir(current_directory)
#         os.mkdir("build")

#         # Execute the commands: cmake .. && make && make install
#         subprocess.check_call(['cmake', '..'])
#         subprocess.check_call(['make'])
#         subprocess.check_call(['make', 'install'])

setup(
    name = "panda_ikfast",
    version = "0.0.1",
    author = "boya",
    author_email = "author.email",
    description = ("Get ik solution of panda robot use ikfast."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license = "",
    packages=find_packages(where='pybind'),
    package_dir={"": "pybind"},
    include_package_data=True,
    package_data={"panda_ikfast":["ikfast_pybind.so"]},
    install_requires=install_deps,
    python_requires='>=3.6',
    # cmdclass={'install': CustomInstallCommand},
    # use entry points to define executable consol scripts 
    # entry_points={'console_scripts': ['script_name=package_name.file_name:main', ], },
)
