from setuptools import find_packages, setup

setup(
    name="dynamicAll",
    packages= ["dynamicAll"],
    package_data = ['src/data'],
    package_dir={"": "dynamicAll"},
)