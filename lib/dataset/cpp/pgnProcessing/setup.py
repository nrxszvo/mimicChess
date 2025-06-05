from setuptools import setup, Extension
from Cython.Build import cythonize
import os
import sys

# Get the directory containing this setup.py file
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
build_dir = os.path.join(project_root, "build")

# Library and include directories
if sys.platform == "darwin":
    system_lib_dir = "/opt/homebrew/lib"  # System libraries
    system_include_dir = "/opt/homebrew/include"  # System includes
else:
    system_lib_dir = "/usr/lib"  # System libraries
    system_include_dir = "/usr/include"  # System includes

extension = Extension(
    "parser_pool",
    sources=["parser_pool.pyx"],
    include_dirs=[
        current_dir,  # For parserPool.h
        project_root,  # For cpp headers
        system_include_dir,  # System headers
    ],
    libraries=[
        "pgnProcessing",  # Our main library
        "arrow",  # Arrow dependency
        "parquet",  # Parquet dependency
        "curl",  # Curl dependency
        "zstd",  # Zstd dependency
        "re2",  # RE2 dependency
        "utils",  # Our utils library
    ],
    library_dirs=[
        os.path.join(build_dir, "pgnProcessing"),  # For libpgnProcessing
        os.path.join(build_dir, "utils"),  # For libutils
        system_lib_dir,  # For system libraries
    ],
    runtime_library_dirs=[
        os.path.join(build_dir, "pgnProcessing"),  # For libpgnProcessing
        os.path.join(build_dir, "utils"),  # For libutils
        system_lib_dir,  # For system libraries
    ],
    language="c++",
    extra_compile_args=["-std=c++20"],
    extra_link_args=[
        "-Wl,-rpath,@loader_path",  # Look for libraries in same dir as module
        "-Wl,-rpath,@loader_path/../../build/pgnProcessing",  # Look in build dir
        "-Wl,-rpath,@loader_path/../../build/utils",  # Look in build dir
    ],
)

setup(
    name="parser_pool",
    ext_modules=cythonize([extension], language_level=3),
)
