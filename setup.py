from setuptools import setup

# Version
version = None
with open("hive_networkx/__init__.py", "r") as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"')
assert version is not None, "Check version in hive_networkx/__init__.py"

setup(
name='hive_networkx',
    version=version,
    description='Hive plots in Python',
    url='https://github.com/jolespin/hive_networkx',
    author='Josh L. Espinoza',
    author_email='jespinoz@jcvi.org',
    license='BSD-3',
    packages=["hive_networkx"],
    install_requires=[
        "pandas >= 1",
        "numpy",
        'scipy >= 1',
        "networkx >= 2",
        "matplotlib >= 3",
        "soothsayer_utils >= 2022.6.24",
        "ensemble_networkx >= 2023.9.25",
      ],
)
