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
@pytest.mark.parametrize("error_distance", [0, 1, 2, 3, 4, 5])
def test_merge_in_radiuses_happy(only_merge, error_distance):
    """Test merge_in_radiuses() with happy path.

    For simplicity, there are no radius sources that are not matched to a target.

    Each target is matched to a radius source, and the radius value is merged.

    """
    radii_sources = [x for x in RADII_SOURCES if x["uuid"] != "7"]
    result = automark.merge_in_radiuses(TARGETS, radii_sources, error_distance, only_merge)

    assert len(result) == 3
    assert result[0]["uuid"] == "1"
    assert result[0]["radius"] == 7
    assert result[1]["uuid"] == "2"
    assert result[1]["radius"] == 12
    assert result[2]["uuid"] == "3"
    assert result[2]["radius"] == 18


@pytest.mark.parametrize("only_merge", [True, False])
@pytest.mark.parametrize("error_distance", [0, 1, 2, 3, 4, 5])
def test_merge_in_radiuses_happy_merge_extra(only_merge, error_distance):
    """Test merge_in_radiuses() with happy path and extra radius sources.

    There is one radius source that is not matched to a target.

    It is either included or not included in the result, depending on only_merge.

    """
    result = automark.merge_in_radiuses(TARGETS, RADII_SOURCES, error_distance, only_merge)

    if only_merge:
        assert len(result) == 3
    else:
        assert len(result) == 4
    assert result[0]["uuid"] == "1"
    assert result[0]["radius"] == 7
    assert result[1]["uuid"] == "2"
    assert result[1]["radius"] == 12
    assert result[2]["uuid"] == "3"
    assert result[2]["radius"] == 18
    if not only_merge:
        assert result[3]["uuid"] == "7"
        assert result[3]["radius"] == 24
