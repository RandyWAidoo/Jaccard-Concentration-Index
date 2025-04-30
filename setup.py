from setuptools import setup, find_packages

setup(
    name='jaccard-concentration-index',
    version='1.0.4',
    description=(
        "A library for clustering evaluation based on the "
        + "distribution of predicted cluster mass across true clusters."
    ),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Randolph Wiredu-Aidoo',
    author_email='randyaidoo.dev@gmail.com',
    license='MIT',
    url="https://github.com/RandyWAidoo/Jaccard-Concentration-Index",
    packages=find_packages(include=['jaccard_concentration_index']),
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