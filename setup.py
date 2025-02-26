#!/usr/bin/env python

from setuptools import setup, find_packages
import os

# https://github.com/ninjaaron/fast-entry_points
import fastentrypoints

with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'),
    encoding='utf-8',
) as readme_md:
    long_description = readme_md.read()

extras_require = {
    'develop': ['bumpversion', 'black', 'pyflakes'],
    'plotting': ['matplotlib'],
    'test': ['pytest', 'pytest-cov', 'coverage', 'pytest-mock'],
    'unfolding': [
        'RooUnfold @ git+https://gitlab.cern.ch/scipp/collinear-w/RooUnfold.git@bug_fixed/dimension_normalizaion'
    ],
    'docs': ['sphinx', 'sphinx_rtd_theme'],
}
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))


setup(
    name='collinearw',
    version='1.5.0',
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["tests"]),
    include_package_data=True,
    description='Python analysis for collinear W',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.cern.ch/scipp/collinearw',
    author='SCIPP',
    author_email='scipp@cern.ch',
    license='',
    keywords='collinear W analysis',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.8, <3.14",
    install_requires=[
        'click',
        'formulate',
        'tabulate',
        'numexpr',
        'numpy',
        'uproot',
        'awkward',
        'tqdm',
        'pandas',
        'pyhf[minuit]',
        'klepto',
        'jsonnet@https://github.com/google/jsonnet/zipball/master',
        'rich',
        'aiofiles',
        'numba',
        'lazy_loader',
    ],
    extras_require=extras_require,
    dependency_links=[],
    entry_points={'console_scripts': ['collinearw=collinearw.cli:collinearw']},
)
