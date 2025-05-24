import setuptools.command.easy_install

# -----------------------------------------------------------------------------
# Monkey-patch setuptools for performance, to avoid importing pkg_resources.
#
# This workaround is copied from BuildStream:
# https://gitlab.com/BuildStream/buildstream/blob/master/setup.py#L143
#
# That solution was inspired by https://github.com/ninjaaron/fast-entry_points
# which we believe was also inspired from the code from `setuptools` project.
#
TEMPLATE = '''\
# -*- coding: utf-8 -*-
import sys

from {0} import {1}

if __name__ == '__main__':
    sys.exit({2}())'''


@classmethod
def get_args(cls, dist, header=None):
    if header is None:
        header = cls.get_header()
    for name, ep in dist.get_entry_map('console_scripts').items():
        cls._ensure_safe_name(name)
        script_text = TEMPLATE.format(ep.module_name, ep.attrs[0], '.'.join(ep.attrs))
        args = cls._get_script_args('console', name, header, script_text)
        for res in args:
            yield res


setuptools.command.easy_install.ScriptWriter.get_args = get_args
# -----------------------------------------------------------------------------

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
        'anthropic',
        'colorama',
        'opencv-python',
        'pandas',
        'pygame',
        'pytorch-lightning',
        'torch',
        'torchvision',
        'tqdm',
        'wandb',
    ],
    extras_require={
        'dev': [
            'black',
            'docformatter',
            'isort',
            'pycodestyle',
            'pyflakes',
            'pylint',
            'pytest',
            'vulture',
        ]
    },
    python_requires='>=3.8',
)
