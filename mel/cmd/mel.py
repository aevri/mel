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
import mel.cmd.rotomapfiltermarkspretrain  # noqa: E402
import mel.cmd.rotomapfiltermarkstrain  # noqa: E402
import mel.cmd.rotomapidentify  # noqa: E402
import mel.cmd.rotomapidentify2  # noqa: E402
import mel.cmd.rotomapidentify2train  # noqa: E402
import mel.cmd.rotomaplist  # noqa: E402
import mel.cmd.rotomaploadsave  # noqa: E402
import mel.cmd.rotomapmarkunchanged  # noqa: E402
import mel.cmd.rotomapmontagesingle  # noqa: E402
import mel.cmd.rotomaporganise  # noqa: E402
import mel.cmd.rotomaprm  # noqa: E402
import mel.cmd.rotomapudiff  # noqa: E402
import mel.cmd.rotomapuuid  # noqa: E402
import mel.cmd.status  # noqa: E402


COMMANDS = {
    "root": {
        "status": mel.cmd.status,
    },
    "micro": {
        "add-cluster": mel.cmd.addcluster,
        "add-single": mel.cmd.addsingle,
        "list": mel.cmd.list,
        "add": mel.cmd.microadd,
        "compare": mel.cmd.microcompare,
        "view": mel.cmd.microview,
    },
    "rotomap": {
        "automark": mel.cmd.rotomapautomark,
        "automask": mel.cmd.rotomapautomask,
        "calc-space": mel.cmd.rotomapcalcspace,
        "compare": mel.cmd.rotomapcompare,
        "confirm": mel.cmd.rotomapconfirm,
        "edit": mel.cmd.rotomapedit,
        "filter-marks": mel.cmd.rotomapfiltermarks,
        "filter-marks-pretrain": mel.cmd.rotomapfiltermarkspretrain,
        "filter-marks-train": mel.cmd.rotomapfiltermarkstrain,
        "identify": mel.cmd.rotomapidentify,
        "identify2": mel.cmd.rotomapidentify2,
        "identify2-train": mel.cmd.rotomapidentify2train,
        "list": mel.cmd.rotomaplist,
        "loadsave": mel.cmd.rotomaploadsave,
        "mark-unchanged": mel.cmd.rotomapmarkunchanged,
        "montage-single": mel.cmd.rotomapmontagesingle,
        "organise": mel.cmd.rotomaporganise,
        "rm": mel.cmd.rotomaprm,
        "udiff": mel.cmd.rotomapudiff,
        "uuid": mel.cmd.rotomapuuid,
    },
}


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    top_subparsers = parser.add_subparsers()

    micro_parser = top_subparsers.add_parser(
        "micro", help="Work with microscope images.", aliases=["m"]
    )

    rotomap_parser = top_subparsers.add_parser(
        "rotomap", help="Work with rotomap images.", aliases=["r", "roto"]
    )

    micro_subparsers = micro_parser.add_subparsers()
    rotomap_subparsers = rotomap_parser.add_subparsers()

    subparsers = top_subparsers

    # Work around a bug in argparse with subparsers no longer being required:
    # http://bugs.python.org/issue9253#msg186387
    subparsers.required = True
    subparsers.dest = "command"

    # vulture will report these as unused unless we do this
    #
    # pylint: disable=pointless-statement
    subparsers.required
    subparsers.dest
    # pylint: enable=pointless-statement

    parser_map = {
        "root": subparsers,
        "micro": micro_subparsers,
        "rotomap": rotomap_subparsers,
    }

    for pname, parser2 in parser_map.items():
        for name, module in COMMANDS[pname].items():
            _setup_parser_for_module(parser2, module, name)

    args = parser.parse_args()
    try:
        return args.func(args)
    except mel.cmd.error.UsageError as e:
        print("Usage error:", e, file=sys.stderr)
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
    doc_epilog = "\n".join(doc.splitlines()[1:])
    parser = subparsers.add_parser(
        name,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help=doc_subject,
        description=doc_subject,
        epilog=doc_epilog,
    )
    module.setup_parser(parser)
    parser.set_defaults(func=module.process_args)


if __name__ == "__main__":
    sys.exit(main())


# -----------------------------------------------------------------------------
# Copyright (C) 2015-2019 Angelos Evripiotis.
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
