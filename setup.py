from setuptools import find_packages, setup

setup(
    name="weird-Jeans",
    packages= ["ndjeans"],
    package_data = ['src/data'],
    package_dir={"": "ndjeans"},
)d