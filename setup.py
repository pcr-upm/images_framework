from setuptools import setup, find_packages

setup(
    name='images_framework',
    version='0.1.0',
    packages=find_packages(include=['src', 'src.*', 'test', 'test.*', 'categories', 'categories.*']),
    install_requires=[
        'opencv-python',
        'opencv-contrib-python',
        'rasterio',
    ],
    author='Roberto Valle',
    author_email='roberto.valle@upm.es',
    description='A modular computer vision framework developed by PCR-UPM for image processing tasks.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/pcr-upm/images_framework',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
