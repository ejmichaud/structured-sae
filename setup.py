from setuptools import setup, find_packages

setup(
    name="structured_sae",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'torch'
    ],
    extras_require={
        "dev": [
            "pytest",
            # Other development dependencies
        ],
    },
)
