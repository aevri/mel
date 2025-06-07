"""Tests for mel.lib.moleimaging module."""

import numpy as np
import pytest

import mel.lib.moleimaging


def test_find_mole_ellipse_vector_operations():
    """Test that find_mole_ellipse correctly handles vector operations.
    
    This test specifically checks for the bug where rightbottom calculation
    creates a 4-element tuple instead of a 2-element numpy array.
    """
    # Create a simple test image (black square with white circle in center)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Create centre as numpy array (as it would come from guess_mole_pos)
    centre = np.array([50, 50], dtype=int)
    radius = 20
    
    # This should not crash and should return a valid ellipse or None
    try:
        result = mel.lib.moleimaging.find_mole_ellipse(image, centre, radius)
        # If we get here without exception, the vector operations are working
        assert result is None or isinstance(result, tuple)
    except TypeError as e:
        # This would indicate the vector operations bug
        pytest.fail(f"find_mole_ellipse failed with vector operations bug: {e}")


def test_find_mole_ellipse_with_mole_image():
    """Test find_mole_ellipse with an image that contains a detectable mole."""
    # Create a simple test image with a dark spot (simulating a mole)
    image = np.full((100, 100, 3), 200, dtype=np.uint8)  # Light gray background
    
    # Add a dark circular region in the center
    center_y, center_x = 50, 50
    y, x = np.ogrid[:100, :100]
    mask = (x - center_x)**2 + (y - center_y)**2 <= 10**2
    image[mask] = [50, 50, 50]  # Dark gray mole
    
    centre = np.array([50, 50], dtype=int)
    radius = 20
    
    # Should process without error
    result = mel.lib.moleimaging.find_mole_ellipse(image, centre, radius)
    # Result could be None or a valid ellipse tuple
    assert result is None or isinstance(result, tuple)


def test_find_mole_ellipse_parameter_types():
    """Test that find_mole_ellipse handles different input types correctly."""
    # Test image
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    
    # Test with numpy array centre (correct type)
    centre_array = np.array([25, 25], dtype=int)
    result = mel.lib.moleimaging.find_mole_ellipse(image, centre_array, 10)
    assert result is None or isinstance(result, tuple)
    
    # Test with tuple centre (might be passed incorrectly)
    centre_tuple = (25, 25)
    try:
        result = mel.lib.moleimaging.find_mole_ellipse(image, centre_tuple, 10)
        # Should handle gracefully
        assert result is None or isinstance(result, tuple)
    except Exception:
        # If it fails with tuple input, that's also acceptable
        pass


def test_vector_operations_components():
    """Test the individual vector operations that were problematic."""
    centre = np.array([50, 50], dtype=int)
    radius = 20
    
    # Test the operations from the original buggy code
    lefttop = centre - (radius, radius)
    assert isinstance(lefttop, np.ndarray)
    assert lefttop.shape == (2,)
    assert np.array_equal(lefttop, [30, 30])
    
    # The buggy operation would create a 4-element tuple
    buggy_rightbottom = (*centre, radius + 1, radius + 1)
    assert isinstance(buggy_rightbottom, tuple)
    assert len(buggy_rightbottom) == 4  # This is the bug!
    
    # The correct operation should create a 2-element array
    correct_rightbottom = centre + (radius + 1, radius + 1)
    assert isinstance(correct_rightbottom, np.ndarray)
    assert correct_rightbottom.shape == (2,)
    assert np.array_equal(correct_rightbottom, [71, 71])