from setuptools import setup, find_packages
from facevision import __version__, __author__, __description__


setup(
    name='facevision',
    version=__version__,
    description=__description__,
    long_description=open('README.md').read(),
    author=__author__,
    author_email='j.mokhtari@itgroup.org',
    packages=find_packages(),
    python_requires='>=3.12',
    # package_data={'facevision': ['assets/weights/*', 'sixdrepnet/*']},
    include_package_data=True,
    # install_requires=['numpy==1.26.4', 'pillow==10.3.0', 'matplotlib==3.9.0', 'scipy==1.13.1',
    #                   'opencv-python==4.10.0.84', 'mediapipe==0.10.14', 'torch==2.3.1', 'torchvision==0.18.1']
)
