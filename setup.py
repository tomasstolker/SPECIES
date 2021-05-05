#!/usr/bin/env python

from setuptools import setup

from pip._internal.network.session import PipSession
from pip._internal.req import parse_requirements

reqs = parse_requirements('requirements.txt', session=PipSession())
reqs = [str(req.requirement) for req in reqs]

setup(
    name='species',
    version='0.3.6',
    description='Toolkit for atmospheric characterization of directly imaged exoplanets',
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
    author='Tomas Stolker',
    author_email='stolker@strw.leidenuniv.nl',
    url='https://github.com/tomasstolker/species',
    project_urls={'Documentation': 'https://species.readthedocs.io'},
    packages=['species',
              'species.analysis',
              'species.core',
              'species.data',
              'species.plot',
              'species.read',
              'species.util'],
    package_dir={'species': 'species'},
    include_package_data=True,
    install_requires=reqs,
    license='MIT',
    zip_safe=False,
    keywords='species',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
