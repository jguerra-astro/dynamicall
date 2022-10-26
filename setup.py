from setuptools import find_packages, setup

setup(
    name="weird-Jeans",
    # packages=find_packages(where="src"),
    packages= ["ndjeans"],
    package_dir={"": "src"},
)