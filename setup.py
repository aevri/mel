import setuptools

lowerbound = setuptools.Extension(
    'mel.rotomap.lowerbound',
    sources=['lowerbound.cpp'],
    extra_compile_args=['-std=c++11'],
)

setuptools.setup(
    name='mel',
    url='https://github.com/aevri/mel',
    author='Angelos Evripiotis',
    author_email='angelos.evripiotis@gmail.com',
    zip_safe=False,
    packages=[
        'mel',
        'mel.cmd',
        'mel.cmddebug',
        'mel.lib',
        'mel.micro',
        'mel.rotomap',
    ],
    entry_points={
        'console_scripts': [
            'mel=mel.cmd.mel:main',
            'mel-debug=mel.cmddebug.meldebug:main',
        ]
    },
    install_requires=[
        # 'opencv',  # Not a possibility as yet.
        'scipy'
    ],
    extras_require={
        'dev': [
            'autopep8',
            'docformatter',
            'flake8',
            'nose',
            'vulture',
            'pylint',
        ]
    },
    ext_modules=[lowerbound],
    python_requires='>=3.6',
)
