from setuptools import setup, find_packages

setup(
    name='plotly_utility',
    version='1.7',
    description='',
    author='yomura',
    author_email='yomura@hoge.jp',
    url='https://github.com/yomura-yomura/plotly_utlility',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "plotly >= 4.14.2",
        "numpy_utility @ git+https://github.com/yomura-yomura/numpy_utility",
    ]
)
