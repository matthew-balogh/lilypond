from setuptools import setup, find_packages

setup(
    name='lilypond',
    version='0.0.1',
    description='Intuitive visualization tool for high-dimensional data using Self-Organizing Maps.',
    long_description=open('USAGE.md').read(),
    long_description_content_type='text/markdown',
    author='Matthew Balogh',
    author_email='matebalogh@inf.elte.hu',
    packages=find_packages(),
    install_requires=[
        "pytest",
        "numpy",
        "matplotlib",
        "plotly",
        "minisom",
    ],
    license='Apache 2.0',
    python_requires='>=3.7',
)
