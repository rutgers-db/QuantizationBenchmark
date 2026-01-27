from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release'
        ]

        build_args = ['--config', 'Release', '-j4']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Copy CMakeLists_pybind.txt to CMakeLists.txt for building
        src_cmake = os.path.join(ext.sourcedir, 'hnsw', 'CMakeLists_pybind.txt')
        dst_cmake = os.path.join(self.build_temp, 'CMakeLists.txt')

        # Create a temporary CMakeLists.txt
        with open(src_cmake, 'r') as f:
            cmake_content = f.read()
        with open(dst_cmake, 'w') as f:
            f.write(cmake_content)

        # Copy source files to build directory
        import shutil
        src_dir = os.path.join(ext.sourcedir, 'hnsw')
        for item in ['hvs_binding.cpp', 'hnswlib']:
            src_path = os.path.join(src_dir, item)
            dst_path = os.path.join(self.build_temp, item)
            if os.path.isdir(src_path):
                if os.path.exists(dst_path):
                    shutil.rmtree(dst_path)
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

        subprocess.check_call(['cmake', self.build_temp] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='hvs',
    version='0.1.0',
    author='HVS Authors',
    description='Hierarchical Vector Search with Python bindings',
    long_description='',
    ext_modules=[CMakeExtension('hvs', sourcedir='.')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires='>=3.6',
)
