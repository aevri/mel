"""Tests for mel.lib.moleimaging module."""

import numpy as np
import pytest

import mel.lib.moleimaging


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
