from setuptools import setup, find_packages

setup(
    name='jaccard_concentration_index',
    version='0.1.0',
    description=(
        "A library for clustering evaluation based on the "
        + "concentration of predicted cluster mass across true clusters."
    ),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Randolph Wiredu-Aidoo',
    author_email='forworkemail914@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'scikit-learn>=0.24.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)