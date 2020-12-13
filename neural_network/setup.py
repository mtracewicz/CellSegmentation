from setuptools import setup

setup(
   name='neural_network',
   version='0.1',
   description='U-Net implementation in Keras',
   author='Michał Tracewicz',
   author_email='m.tracewicz@gmail.com',
   packages=['unet'],
   install_requires=['numpy', 'Pillow', 'progress','tensorflow', 'keras']
)