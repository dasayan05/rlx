import setuptools

setuptools.setup(
    name = 'rlx',
    version = '0.1',
    author = 'Ayan Das',
    author_email = 'a.das@surrey.ac.uk',
    description = 'A modular and generic Reinforcement Learning (RL) library for research',
    packages = ['rlx'],
    license = 'MIT License',
    keywords = 'reinforcement learning deep pytorch',
    install_requires = ['torch'],
    classifiers = [
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research'
    ]
)
