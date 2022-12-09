from setuptools import setup, find_packages

def load_requires_from_file(filepath):
    with open(filepath) as fp:
        return [pkg_name.strip() for pkg_name in fp.readlines()]

setup(
    name='tmanual',
    version='0.1',
    description='tool to assist in measuring length development of structures',
    author='Nobuaki Mizumoto',
    author_email="nobuaki.mzmt@gmail.com",
    license='BSD-3',
    packages=find_packages(),
    install_requires=load_requires_from_file("requirements.txt")
)