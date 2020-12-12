from setuptools import setup

setup(
   name='image_preprocessing',
   version='0.1',
   description='Image preprocessing module for unet',
   author='Michał Tracewicz',
   author_email='m.tracewicz@gmail.com',
   packages=['preprocessing'],
   install_requires=['numpy', 'Pillow', 'progress']
)