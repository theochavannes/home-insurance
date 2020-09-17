from setuptools import setup, find_packages

DISTNAME = 'home-insurance'
DESCRIPTION = 'A library to solve the home insurance dataset problem'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()


def parse_requirements(req_file):
    with open(req_file) as fp:
        _requires = fp.read()
    return _requires


# Get dependencies from requirement files
SETUP_REQUIRES = ['setuptools', 'setuptools-git', 'wheel']
INSTALL_REQUIRES = parse_requirements('requirements.txt')


def setup_package():
    metadata = dict(name=DISTNAME,
                    long_description=LONG_DESCRIPTION,
                    long_description_content_type="text/markdown",
                    install_requires=INSTALL_REQUIRES,
                    setup_requires=SETUP_REQUIRES,
                    packages=find_packages(include=["home_insurance*"]))

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
