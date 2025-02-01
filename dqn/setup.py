# setup.py
import os
import sys
import pybind11
from setuptools import setup, Extension

extra_args = []
# Windows MSVC typical
if os.name == 'nt':
    extra_args = ['/std:c++14','/O2']
else:
    extra_args = ['-std=c++14','-O3']

ext_modules = [
    Extension(
        'mcts_cpp', 
        ['mcts.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=extra_args
    ),
]

setup(
    name='mcts_cpp',
    version='1.0',
    author='TomSawyer2',
    ext_modules=ext_modules,
    description='AlphaZero MCTS for 5x5 on Windows + MSVC fix'
)
