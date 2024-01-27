"""Test suite for mel.rotomap.relate."""

import pytest

import mel.rotomap.automark as automark

TARGETS = [
    {"uuid": "1", "x": 0, "y": 0},
    {"uuid": "2", "x": 1, "y": 1},
    {"uuid": "3", "x": 2, "y": 2}
]

RADII_SOURCES = [
    {"uuid": "4", "radius": 7, "x": 0, "y": 0},
    {"uuid": "5", "radius": 12, "x": 1, "y": 1},
    {"uuid": "6", "radius": 18, "x": 2, "y": 2},
    {"uuid": "7", "radius": 24, "x": 3, "y": 3},
]


@pytest.mark.parametrize("only_merge", [True, False])
def test_merge_in_radiuses_happy(only_merge):
    radii_sources = [x for x in RADII_SOURCES if x["uuid"] != "7"]

    error_distance = 3

    result = automark.merge_in_radiuses(TARGETS, radii_sources, error_distance, only_merge)

    assert len(result) == 3
    assert result[0]["uuid"] == "1"
    assert result[0]["radius"] == 7
    assert result[1]["uuid"] == "2"
    assert result[1]["radius"] == 12
    assert result[2]["uuid"] == "3"
    assert result[2]["radius"] == 18


def test_merge_in_radiuses_happy_only_merge():
    error_distance = 3
    only_merge = True

    result = automark.merge_in_radiuses(TARGETS, RADII_SOURCES, error_distance, only_merge)

    assert len(result) == 3
    assert result[0]["uuid"] == "1"
    assert result[0]["radius"] == 7
    assert result[1]["uuid"] == "2"
    assert result[1]["radius"] == 12
    assert result[2]["uuid"] == "3"
    assert result[2]["radius"] == 18


def test_merge_in_radiuses_happy_not_only_merge():
    error_distance = 3
    only_merge = False

    result = automark.merge_in_radiuses(TARGETS, RADII_SOURCES, error_distance, only_merge)

    assert len(result) == 4
    assert result[0]["uuid"] == "1"
    assert result[0]["radius"] == 7
    assert result[1]["uuid"] == "2"
    assert result[1]["radius"] == 12
    assert result[2]["uuid"] == "3"
    assert result[2]["radius"] == 18
    assert result[3]["uuid"] == "7"
    assert result[3]["radius"] == 24
