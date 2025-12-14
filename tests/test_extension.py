"""
Unit tests for Example Extension
"""

import pytest
from medics_ext_example import ExampleExtension


class MockContext:
    """Mock application context for testing."""
    
    def __init__(self):
        self.components = {}
    
    def get_component(self, name):
        return self.components.get(name)


def test_extension_metadata():
    """Test extension metadata methods."""
    ext = ExampleExtension()
    
    assert ext.get_name() == "Example Extension"
    assert ext.get_author() == "Your Name"
    assert ext.get_version() == "1.0.0"
    assert ext.get_category() == "Examples"
    assert len(ext.get_description()) > 0


def test_extension_initialize():
    """Test extension initialization."""
    ext = ExampleExtension()
    context = MockContext()
    
    result = ext.initialize(context)
    
    assert result is True
    assert ext.app_context is context


def test_extension_create_widget():
    """Test widget creation."""
    ext = ExampleExtension()
    context = MockContext()
    ext.initialize(context)
    
    widget = ext.create_widget()
    
    assert widget is not None
    assert ext.widget is widget


def test_extension_cleanup():
    """Test extension cleanup."""
    ext = ExampleExtension()
    context = MockContext()
    ext.initialize(context)
    
    # Should not raise exception
    ext.cleanup()
    
    assert ext.widget is None


if __name__ == "__main__":
    pytest.main([__file__])
