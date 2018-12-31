"""Mel - a command-line utility to help with mole management."""


import argparse
import sys


OLD_PYTHON_MESSAGE = """You are running mel on an old Python interpreter

Unfortunately Frobulator 6.0 and above are not compatible with Python < 3.6
anymore, and you still ended up with this version installed on your system.
Make sure you have pip >= 9.0 to avoid this kind of issues, as well as
setuptools >= 24.2:

 $ pip install pip setuptools --upgrade

Upgrade your system to use Python 3.6 or later to run this version of mel.

"""

if sys.version_info < (3, 6):
    raise ImportError(OLD_PYTHON_MESSAGE)


# note that we do this to ignore warnings about top-level imports not being at
# the top of the file - noqa: E402

import mel.cmd.addcluster  # noqa: E402
import mel.cmd.addsingle  # noqa: E402
import mel.cmd.error  # noqa: E402
import mel.cmd.list  # noqa: E402
import mel.cmd.microadd  # noqa: E402
import mel.cmd.microcompare  # noqa: E402
import mel.cmd.microview  # noqa: E402
import mel.cmd.rotomapautomark  # noqa: E402
import mel.cmd.rotomapautomask  # noqa: E402
import mel.cmd.rotomapcalcspace  # noqa: E402
import mel.cmd.rotomapcompare  # noqa: E402
import mel.cmd.rotomapconfirm  # noqa: E402
import mel.cmd.rotomapedit  # noqa: E402
import mel.cmd.rotomapfiltermarks  # noqa: E402
import mel.cmd.rotomapidentify  # noqa: E402
import mel.cmd.rotomaplist  # noqa: E402
import mel.cmd.rotomaploadsave  # noqa: E402
import mel.cmd.rotomapmontagesingle  # noqa: E402
import mel.cmd.rotomaporganise  # noqa: E402
import mel.cmd.rotomaprm  # noqa: E402
import mel.cmd.rotomapudiff  # noqa: E402
import mel.cmd.rotomapuuid  # noqa: E402
import mel.cmd.status  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    subparsers = parser.add_subparsers()

    # Work around a bug in argparse with subparsers no longer being required:
    # http://bugs.python.org/issue9253#msg186387
    subparsers.required = True
    subparsers.dest = 'command'

    # vulture will report these as unused unless we do this
    #
    # pylint: disable=pointless-statement
    subparsers.required
    subparsers.dest
    # pylint: enable=pointless-statement

    _setup_parser_for_module(subparsers, mel.cmd.addcluster, 'add-cluster')
    _setup_parser_for_module(subparsers, mel.cmd.addsingle, 'add-single')
    _setup_parser_for_module(subparsers, mel.cmd.list, 'list')
    _setup_parser_for_module(subparsers, mel.cmd.microadd, 'micro-add')
    _setup_parser_for_module(subparsers, mel.cmd.microcompare, 'micro-compare')
    _setup_parser_for_module(subparsers, mel.cmd.microview, 'micro-view')
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomapautomark, 'rotomap-automark'
    )
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomapautomask, 'rotomap-automask'
    )
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomapcalcspace, 'rotomap-calc-space'
    )
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomapcompare, 'rotomap-compare'
    )
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomapconfirm, 'rotomap-confirm'
    )
    _setup_parser_for_module(subparsers, mel.cmd.rotomapedit, 'rotomap-edit')
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomapfiltermarks, 'rotomap-filter-marks'
    )
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomapidentify, 'rotomap-identify'
    )
    _setup_parser_for_module(subparsers, mel.cmd.rotomaplist, 'rotomap-list')
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomaploadsave, 'rotomap-loadsave'
    )
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomapmontagesingle, 'rotomap-montage-single'
    )
    _setup_parser_for_module(
        subparsers, mel.cmd.rotomaporganise, 'rotomap-organise'
    )
    _setup_parser_for_module(subparsers, mel.cmd.rotomaprm, 'rotomap-rm')
    _setup_parser_for_module(subparsers, mel.cmd.rotomapudiff, 'rotomap-udiff')
    _setup_parser_for_module(subparsers, mel.cmd.rotomapuuid, 'rotomap-uuid')
    _setup_parser_for_module(subparsers, mel.cmd.status, 'status')

    args = parser.parse_args()
    try:
        return args.func(args)
    except mel.cmd.error.UsageError as e:
        print('Usage error:', e, file=sys.stderr)
        return 2
    except BrokenPipeError:
        # Silently exit on broken pipes, e.g. when our output is piped to head.

        # Explicitly close stderr before exiting, to avoid an additional
        # message from Python on stderr about the pipe break being ignored.
        # http://bugs.python.org/issue11380,#msg153320
        sys.stderr.close()
    except mel.lib.ui.AbortKeyInterruptError:
        # Using this return code may also break us out of an outer loop, e.g.
        # 'xargs' will stop processing if the program it calls exists with 255.
        return 255


def _setup_parser_for_module(subparsers, module, name):
    doc = module.__doc__
    doc_subject = doc.splitlines()[0]
    doc_epilog = '\n'.join(doc.splitlines()[1:])
    parser = subparsers.add_parser(
        name,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help=doc_subject,
        description=doc_subject,
        epilog=doc_epilog,
    )
    module.setup_parser(parser)
    parser.set_defaults(func=module.process_args)


if __name__ == '__main__':
    sys.exit(main())
# -----------------------------------------------------------------------------
# Copyright (C) 2015-2018 Angelos Evripiotis.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------ END-OF-FILE ----------------------------------
