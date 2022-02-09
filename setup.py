from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="cairo_planning_core",
    version="1.0",
    rust_extensions=[RustExtension("cairo_planning_core", binding=Binding.PyO3)],
    packages=["cairo_planning_core"],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)