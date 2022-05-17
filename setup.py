from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

setup(
      name='neyboy',
      packages=[package for package in find_packages()],
      install_requires=[
          'stable-baselines3[extra]',
          'gym-neyboy@git+https://github.com/fabito/gym-neyboy.git#egg=gym-neyboy'
      ],
      description='Neyboy Challenge AI Agent.',
      author='Fabito',
      url='https://github.com/fabito/neyboy',
      author_email='fabio.uechi@gmail.com',
      version='0.2'
)
