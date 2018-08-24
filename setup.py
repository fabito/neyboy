from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

setup(name='neyboy',
      packages=[package for package in find_packages()],
      install_requires=[
          'tensorforce',
      ],
      extras_require={
            "tf": ["tensorflow"],
            "tf_gpu": ["tensorflow-gpu"],
      },
      description='Neyboy Challenge AI Agent.',
      author='Fabito',
      url='https://github.com/fabito/neyboy',
      author_email='fabio.uechi@gmail.com',
      version='0.1')