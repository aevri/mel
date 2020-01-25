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
        'scipy',
        'torch==1.1.0',  # Oddly 1.1.0.post2 is much slower on Mac, so pin.
        'torchvision<0.4.0',  # Match version of torch.
        'tqdm',
        'six>=1.12',  # Oddly we get a version that's too old otherwise.
        'pillow<7',  # https://github.com/pytorch/vision/issues/1712
    ],
    extras_require={
        'dev': [
            'autopep8',
            'docformatter',
            'flake8',
            'pycodestyle',
            'nose',
            'vulture',
            'pylint',
        ]
    },
    ext_modules=[lowerbound],
    python_requires='>=3.6',
)
