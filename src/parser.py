"""
Name parsing module.

This module provides functions for parsing Filipino names, with special handling
for the middle_name_last_name field format commonly used in Filipino identity data.
"""

from typing import Dict, List, Tuple


def parse_name(
    first_name: str, middle_name_last_name: str = None
) -> Dict[str, str]:
    """
    Parse a Filipino name into its components.

    Args:
        first_name: The first name of the person
        middle_name_last_name: The combined middle name and last name field

    Returns:
        A dictionary containing the parsed name components:
        {
            'first_name': str,
            'middle_name': str,
            'last_name': str
        }
    """
    result = {"first_name": first_name.strip(), "middle_name": "", "last_name": ""}

    if not middle_name_last_name:
        return result

    # Handle the middle_name_last_name field
    parts = _split_middle_name_last_name(middle_name_last_name)
    
    if len(parts) == 1:
        # Only last name is present
        result["last_name"] = parts[0]
    elif len(parts) >= 2:
        # Last part is the last name, everything else is middle name
        result["last_name"] = parts[-1]
        result["middle_name"] = " ".join(parts[:-1])

    return result


def _split_middle_name_last_name(middle_name_last_name: str) -> List[str]:
    """
    Split the middle_name_last_name field into components, handling Filipino naming patterns.

    Args:
        middle_name_last_name: The combined middle name and last name field

    Returns:
        A list of name components
    """
    # Clean the input
    name = middle_name_last_name.strip()
    if not name:
        return []

    # Handle compound surnames with special prefixes
    compound_prefixes = ["dela", "de la", "del", "de los", "de las", "san", "santa", "sto", "sta"]
    
    # Split by spaces
    parts = name.split()
    
    # Process parts to handle compound surnames
    processed_parts = []
    i = 0
    while i < len(parts):
        # Check if current part starts a compound surname
        is_compound = False
        for prefix in compound_prefixes:
            prefix_parts = prefix.split()
            if i + len(prefix_parts) <= len(parts):
                potential_prefix = " ".join(parts[i:i+len(prefix_parts)]).lower()
                if potential_prefix == prefix and i + len(prefix_parts) < len(parts):
                    # Found a compound surname, combine with the next part
                    compound = " ".join(parts[i:i+len(prefix_parts)+1])
                    processed_parts.append(compound)
                    i += len(prefix_parts) + 1
                    is_compound = True
                    break
        
        if not is_compound:
            processed_parts.append(parts[i])
            i += 1
    
    return processed_parts


def extract_name_components(full_name: str) -> Dict[str, str]:
    """
    Extract components from a full name string.

    Args:
        full_name: The full name as a single string

    Returns:
        A dictionary containing the parsed name components
    """
    parts = full_name.strip().split()
    
    if not parts:
        return {"first_name": "", "middle_name": "", "last_name": ""}
    
    if len(parts) == 1:
        return {"first_name": parts[0], "middle_name": "", "last_name": ""}
    
    if len(parts) == 2:
        return {"first_name": parts[0], "middle_name": "", "last_name": parts[1]}
    
    # For names with 3 or more parts
    first_name = parts[0]
    last_name = parts[-1]
    middle_name = " ".join(parts[1:-1])
    
    # Check for compound surnames
    return parse_name(first_name, f"{middle_name} {last_name}")
