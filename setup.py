from setuptools import setup

setup(
    name='tddt',
    version='0.3.3',
    url='https://github.com/krivenko/tddt.git',
    author='Igor Krivenko, Viktor Harkov, Viktor Valmispild',
    author_email='iskrivenko@proton.me',
    description='Implementation of the time-dependent dual TRILEX theory',
    packages=['tddt'],
    install_requires=['numpy >= 1.12.0', 'scipy >= 1.7.0'],
)
