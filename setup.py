try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

try:
   import pypandoc
   description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   description = 'Basic implementation of Gaussian mixture model.'

setup(
    name='GMM',
    version='0.1.0',
    author='Jakub Konka',
    author_email='kubkon@gmail.com',
    packages=['gmm', 'gmm.test'],
    url='https://github.com/kubkon/gmm',
    license='LICENSE.txt',
    description='Basic implementation of Gaussian mixture model.',
    long_description=description,
    install_requires=[
        "numpy>=1.7.1",
        "matplotlib>=1.3.0",
    ],
)