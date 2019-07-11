from setuptools import find_packages, setup


setup(
    name='Arbiter',
    version='0.2',
    description='Use machine-learning models to predict malware',
    license='LICENSE',
    packages=find_packages(),
    install_requires=[
        'python-magic',
        'pefile',
        'pandas',
        'scikit-learn',
        'xgboost',
    ],
    entry_points={
        'console_scripts': [
            'arbiter = arbiter.app:cli_app',
        ],
    }
)
