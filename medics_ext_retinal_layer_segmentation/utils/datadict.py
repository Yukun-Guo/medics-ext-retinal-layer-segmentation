"""Dictionary with dot notation access.

This module provides a dictionary subclass that allows attribute-style access
to dictionary items using dot notation.
"""

import logging


# Module logger
logger = logging.getLogger(__name__)


class dataDict(dict):
    """Dictionary subclass that allows dot notation access to dictionary attributes.
    
    This class extends the built-in dict class to provide convenient attribute-style
    access to dictionary items. You can access dictionary values using dot notation
    (e.g., mydict.key) instead of bracket notation (e.g., mydict['key']).
    
    Attributes:
        All dictionary items are accessible as attributes.
        
    Example:
        Basic usage with simple values:
        
        >>> mydict = dataDict({'val': 'it works'})
        >>> mydict.val
        'it works'
        
        Nested dictionaries also work:
        
        >>> mydict = dataDict({'val': 'it works'})
        >>> nested_dict = dataDict({'val': 'nested works too'})
        >>> mydict.nested = nested_dict
        >>> mydict.nested.val
        'nested works too'
        
    Note:
        - Attribute names must be valid Python identifiers
        - Dictionary methods like get(), items(), etc. are still accessible
        - Setting, getting, and deleting attributes maps to dictionary operations
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    @staticmethod
    def from_dict(data):
        """Recursively convert a dictionary and all nested dictionaries to dataDict.
        
        This method walks through the entire dictionary structure and converts
        all dict instances to dataDict instances, enabling dot notation access
        throughout the entire nested structure.
        
        Args:
            data: The data to convert. Can be a dict, list, tuple, or any other type.
            
        Returns:
            The converted data with all dicts replaced by dataDict instances.
            
        Example:
            >>> nested = {'level1': {'level2': {'level3': 'value'}}}
            >>> dd = dataDict.from_dict(nested)
            >>> dd.level1.level2.level3
            'value'
        """
        if isinstance(data, dict) and not isinstance(data, dataDict):
            # Convert the dict to dataDict and recursively convert all values
            return dataDict({key: dataDict.from_dict(value) for key, value in data.items()})
        elif isinstance(data, list):
            # Recursively convert all items in the list
            return [dataDict.from_dict(item) for item in data]
        elif isinstance(data, tuple):
            # Recursively convert all items in the tuple
            return tuple(dataDict.from_dict(item) for item in data)
        else:
            # Return the data as-is (primitives, numpy arrays, etc.)
            return data

# Backwards compatible alias following PEP8
DataDict = dataDict