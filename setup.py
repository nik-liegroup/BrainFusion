from setuptools import setup, find_packages

setup(
    name='spectroscopy_postprocessing',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'h5py >= 3.11.0',
        'matplotlib >= 3.8.4',
        'numpy >= 2.1.1',
        'opencv_python >= 4.10.0.84',
        'pandas >= 2.2.3',
        'Pillow >= 10.4.0',
        'scikit_learn >= 1.4.2',
        'scipy >= 1.14.1',
        'scikit-image>=0.23'
    ],
    author='Niklas Gampl',
    author_email='niklas.gampl@fau.de',
    description='A package for postprocessing Brillouin and AFM spectroscopy data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/nik-liegroup/spectroscopy_postprocessing',  # URL to your repository
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
