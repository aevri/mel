"""Utilities for working with datetime."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os


def guess_datetime_from_path(path):
    """Return None if no date could be guessed, datetime otherwise.

    Usage examples:

        >>> guess_datetime_from_path('inbox/Photo 05-01-2015 23 25 40.jpg')
        datetime.datetime(2015, 1, 5, 23, 25, 40)

        >>> guess_datetime_from_path('20120107T115426.790019.jpg')
        datetime.datetime(2012, 1, 7, 11, 54, 26)

        >>> guess_datetime_from_path('blah')

    :path: path string to be converted
    :returns: datetime.date if successful, None otherwise

    """
    # TODO: try the file date if unable to determine from name
    filename = os.path.basename(path)
    name = filename.split('.', 1)[0]
    return guess_datetime_from_string(name)


def guess_datetime_from_string(datetime_str):
    """Return None if no datetime could be guessed, datetime otherwise.

    Usage examples:

        >>> guess_datetime_from_string('Photo 05-01-2015 23 25 40')
        datetime.datetime(2015, 1, 5, 23, 25, 40)

        >>> guess_datetime_from_string('blah')

    :datetime_str: string to be converted
    :returns: datetime.datetime if successful, None otherwise

    """
    format_list = [
        'Photo %d-%m-%Y %H %M %S',
        '%Y%m%dT%H%M%S',
    ]
    for fmt in format_list:
        try:
            return datetime.datetime.strptime(datetime_str, fmt)
        except ValueError:
            pass
    return None


def make_now_datetime_string():
    return make_datetime_string(datetime.datetime.utcnow())


def make_datetime_string(datetime_):
    return datetime_.strftime("%Y%m%dT%H%M%S")
