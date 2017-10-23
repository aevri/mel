import setuptools

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
    python_requires='>=3.6',
)
