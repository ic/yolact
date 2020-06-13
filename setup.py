from setuptools import setup, find_namespace_packages

setup(
    author='',
    author_email='',
    description='',
    install_requires=[
        'python==^3.7',
        'cython==^0.29.17',
        'opencv-python==^4.2.0',
        'pillow==<7',
        'pycocotools==^2.0.0',
        'matplotlib==^3.2.1',
        'torch==^1.5.0',
        'torchvision==^0.6.0',
    ],
    keywords='',
    name='yolact',
    packages=find_namespace_packages(
        where='src',
        exclude=[]
    ),
    package_data={
        '': [ '*.hdf5' ],
    },
    package_dir={'': 'src'},
    url='',
    version='0.1',
)
