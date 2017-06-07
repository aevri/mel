"""Operate on a 'mel micro' filesystem."""

import collections
import datetime
import pathlib


Mole = collections.namedtuple(
    'mel_micro_fs__Mole',
    [
        'abspath',
        'path',
        'refrelpath',
        'id',
        'need_assistance',
        'context_image_name_tuple_tuple',  # The most local paths appear last
        'micro_image_names',
        'last_micro',
        'last_micro_age_days',
    ]
)


class Names():
    MICRO = '__micro__'
    ID = '__id__'
    NEED_ASSISTANCE = '__need_assistance__'
    CHANGED = '__changed__'
    NOT_CHANGED = '__not_changed__'


FILES_TO_IGNORE = {
    '.DS_Store',
}


DIRS_TO_IGNORE = {
    '.git',
}


IMAGE_SUFFIXES = {
    '.jpg',
    '.jpeg',
    '.png',
}


MOLE_DIR_ENTRIES = {
    Names.MICRO,
    Names.ID,
    Names.NEED_ASSISTANCE,
    Names.CHANGED,
    Names.NOT_CHANGED,
}


def yield_moles(path):
    path = pathlib.Path(path)
    yield from _yield_moles_imp(path, path, ())


def _yield_moles_imp(path, refrelpath, context_image_name_tuple_tuple):
    should_be_mole_dir = False
    for sub in path.iterdir():
        if sub.name.lower() in MOLE_DIR_ENTRIES:
            should_be_mole_dir = True
            break

    context_image_name_tuple_tuple = _extend_context_image_name_tuple_tuple(
        path, context_image_name_tuple_tuple)

    if should_be_mole_dir:
        _validate_mole_dir(path)
        micro_image_names = _list_micro_dir_if_exists(path / Names.MICRO)
        yield Mole(
            abspath=path.resolve(strict=True),
            path=path,
            refrelpath=path.relative_to(refrelpath),
            id=_read_stripped_text_file_if_exists(path / Names.ID),
            need_assistance=(path / Names.NEED_ASSISTANCE).exists(),
            context_image_name_tuple_tuple=context_image_name_tuple_tuple,
            micro_image_names=micro_image_names,
            last_micro=calc_last_micro(micro_image_names),
            last_micro_age_days=calc_last_micro_age_days(micro_image_names),
        )
    else:
        for sub in path.iterdir():
            if sub.is_dir():
                yield from _yield_moles_imp(
                    sub, refrelpath, context_image_name_tuple_tuple)


def _validate_mole_dir(path):
    for sub in path.iterdir():
        if sub.name.lower() not in MOLE_DIR_ENTRIES:

            if sub.suffix.lower() in IMAGE_SUFFIXES:
                continue

            if sub.name in FILES_TO_IGNORE and sub.is_file():
                continue

            if sub.name in DIRS_TO_IGNORE and sub.is_dir():
                continue

            raise Exception('Unexpected in a mole dir: {}'.format(sub))


def _extend_context_image_name_tuple_tuple(
        path, context_image_name_tuple_tuple):

    image_names = []
    for sub in path.iterdir():
        if sub.suffix.lower() in IMAGE_SUFFIXES:
            image_names.append(sub.name)

    if image_names:
        image_names.sort()
        return context_image_name_tuple_tuple + (tuple(image_names), )
    else:
        return context_image_name_tuple_tuple


def _list_micro_dir_if_exists(path):
    if not path.exists():
        return tuple()

    image_names = []
    for sub in path.iterdir():

        if sub.name in FILES_TO_IGNORE and sub.is_file():
            continue

        if sub.name in DIRS_TO_IGNORE and sub.is_dir():
            continue

        if sub.is_dir():
            raise Exception(
                'Sub-directory found in micro dir: {}'.format(
                    sub.resolve()))

        if sub.suffix.lower() not in IMAGE_SUFFIXES:
            raise Exception(
                'Non-image found in micro dir: {}'.format(
                    sub.resolve()))

        image_names.append(sub.name)

    image_names.sort()
    return tuple(image_names)


def calc_last_micro(micro_image_names):
    if not micro_image_names:
        return None

    return micro_image_names[-1]


def calc_last_micro_age_days(micro_image_names):
    now = datetime.datetime.now()

    if not micro_image_names:
        return None

    lastmicro = micro_image_names[-1]
    lastmicrodtstring = lastmicro.split('.', 1)[0]
    lastmicrodt = datetime.datetime.strptime(
        lastmicrodtstring, '%Y%m%dT%H%M%S')
    age = now - lastmicrodt
    return age.days


def _read_stripped_text_file_if_exists(path):
    if path.exists():
        return path.read_text().strip()
    return None