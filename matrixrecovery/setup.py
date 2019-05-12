from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext

modulename = 'myutils_cython'
# extra_compile_args = ["-ffast-math"])
ext_modules=[ Extension(modulename, [modulename+".pyx"], libraries=["m"], extra_compile_args = ["-ffast-math"])]

setup(
#    ext_modules = cythonize("helloworld.pyx")
    name = modulename,
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules
)

