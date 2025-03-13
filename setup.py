from setuptools import setup, find_packages

setup(
    name='brainfusion',
    version='0.1.5',
    packages=find_packages(),
    install_requires=[  # Used for distributing package; requirements.txt for setting up virtual env for developing
        'h5py >= 3.11.0',
        'matplotlib >= 3.8.4',
        'numpy >= 2.1.1',
        'opencv_python >= 4.10.0.84',
        'pandas >= 2.2.3',
        'pyarrow >= 19.0.1',
        'Pillow >= 10.4.0',
        'scikit_learn >= 1.5.2',
        'scipy >= 1.14.1',
        'scikit-image>=0.23',
        'shapely >=2.0.6',
        'frechetdist >= 0.6',
        'tqdm ~= 4.67.1'
    ],
    author='Niklas Gampl',
    author_email='niklas.gampl@mpzpm.mpg.de',
    description='A package for spatially correlating Brillouin and AFM spectroscopy data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/nik-liegroup/BrainFusion',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
