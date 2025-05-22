"""
Name matching module.

This module provides functions for comparing names using various string similarity
algorithms, including Jaro-Winkler, Soundex, and Jaccard Index.
"""

import re
import logging
from typing import Dict, List, Set, Tuple
from numba import jit, cuda
import numpy as np
import jellyfish # Keep jellyfish for Damerau-Levenshtein as primary
from functools import lru_cache

logger = logging.getLogger(__name__)

# Helper for Numba CUDA device selection (optional, but good practice)
CUDA_AVAILABLE = False
try:
    if cuda.is_available():
        cuda.select_device(0) # Selects the first available GPU
        CUDA_AVAILABLE = True
        logger.info("CUDA is available. Numba will use GPU where specified and possible.")
    else:
        logger.info("CUDA not available. Numba will use CPU.")
except Exception as e: # Catch generic Exception as Numba might raise different errors
    logger.warning(f"Error initializing CUDA: {e}. Numba will use CPU.")
    CUDA_AVAILABLE = False


@jit(nopython=True, cache=True)
def _jaro_similarity_numba(s1_arr, s2_arr):
    """
    Numba-jitted Jaro similarity core logic.
    Assumes s1_arr and s2_arr are NumPy arrays of char codes (e.g., uint8 or uint32 for Unicode).
    """
    len_s1, len_s2 = len(s1_arr), len(s2_arr)

    if len_s1 == 0 and len_s2 == 0:
        return 1.0
    if len_s1 == 0 or len_s2 == 0:
        return 0.0

    # Check for exact match first
    if len_s1 == len_s2:
        match = True
        for i in range(len_s1):
            if s1_arr[i] != s2_arr[i]:
                match = False
                break
        if match:
            return 1.0

    match_distance = max(len_s1, len_s2) // 2 - 1
    match_distance = max(0, match_distance)

    s1_matches = np.zeros(len_s1, dtype=np.bool_)
    s2_matches = np.zeros(len_s2, dtype=np.bool_)

    matching_chars = 0
    for i in range(len_s1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_s2)
        for j in range(start, end):
            if not s2_matches[j] and s1_arr[i] == s2_arr[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matching_chars += 1
                break

    if matching_chars == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(len_s1):
        if s1_matches[i]:
            # Find the corresponding matched character in s2_arr
            while k < len_s2 and not s2_matches[k]:
                k += 1
            if k < len_s2 and s1_arr[i] != s2_arr[k]: # Check k < len_s2 to prevent index out of bounds
                transpositions += 1
            k += 1

    transpositions //= 2

    return (
        matching_chars / len_s1
        + matching_chars / len_s2
        + (matching_chars - transpositions) / matching_chars
    ) / 3.0

def _jaro_similarity_python(s1: str, s2: str) -> float:
    """
    Calculate the Jaro similarity between two strings (Original Python version).
    """
    if s1 == s2:
        return 1.0

    len_s1, len_s2 = len(s1), len(s2)

    if len_s1 == 0 or len_s2 == 0:
        return 0.0

    match_distance = max(len_s1, len_s2) // 2 - 1
    match_distance = max(0, match_distance)

    s1_matches = [False] * len_s1
    s2_matches = [False] * len_s2

    matching_chars = 0
    for i in range(len_s1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_s2)
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matching_chars += 1
                break

    if matching_chars == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(len_s1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

    transpositions //= 2

    return (
        matching_chars / len_s1
        + matching_chars / len_s2
        + (matching_chars - transpositions) / matching_chars
    ) / 3.0

_jaro_similarity = _jaro_similarity_python # Default to python version

# Wrapper to select Numba JIT version if available and types are convertible
def jaro_similarity_dispatch(s1: str, s2: str) -> float:
    try:
        # Convert strings to NumPy arrays of Unicode code points (uint32)
        s1_arr = np.array([ord(c) for c in s1], dtype=np.uint32)
        s2_arr = np.array([ord(c) for c in s2], dtype=np.uint32)
        return _jaro_similarity_numba(s1_arr, s2_arr)
    except Exception as e:
        logger.warning(f"Numba JIT for _jaro_similarity failed: {e}. Falling back to Python version.")
        return _jaro_similarity_python(s1, s2)

@lru_cache(maxsize=10000)
def jaro_winkler_similarity(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """
    Calculate the Jaro-Winkler similarity between two strings.
    Uses Numba JIT-compiled Jaro similarity if possible.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"jaro_winkler_similarity called with s1: '{s1}', s2: '{s2}', prefix_weight: {prefix_weight}")

    if not s1 and not s2: # Handle empty strings
        result = 1.0
        logger.debug(f"jaro_winkler_similarity returning (both empty): {result}")
        return result
    if not s1 or not s2: # Handle one empty string
        result = 0.0
        logger.debug(f"jaro_winkler_similarity returning (one empty): {result}")
        return result

    jaro_score = jaro_similarity_dispatch(s1, s2)

    prefix_len = 0
    max_prefix_len = min(4, min(len(s1), len(s2)))
    for i in range(max_prefix_len):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    jaro_winkler_score = jaro_score + (prefix_len * prefix_weight * (1 - jaro_score))
    logger.debug(f"jaro_winkler_similarity for '{s1}' vs '{s2}': jaro_score={jaro_score:.4f}, prefix_len={prefix_len}, result={jaro_winkler_score:.4f}")
    return jaro_winkler_score


@jit(nopython=True, cache=True)
def damerau_levenshtein_distance_numba(s1_arr, s2_arr):
    """
    Numba-jitted Damerau-Levenshtein distance (Optimal String Alignment variant).
    Assumes s1_arr and s2_arr are NumPy arrays of char codes (e.g., uint32).
    """
    len_str1 = len(s1_arr)
    len_str2 = len(s2_arr)
    d = np.zeros((len_str1 + 1, len_str2 + 1), dtype=np.int32)

    for i in range(len_str1 + 1):
        d[i, 0] = i
    for j in range(len_str2 + 1):
        d[0, j] = j

    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            cost = 0 if s1_arr[i - 1] == s2_arr[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1,          # Deletion
                          d[i, j - 1] + 1,          # Insertion
                          d[i - 1, j - 1] + cost)   # Substitution
            if i > 1 and j > 1 and s1_arr[i-1] == s2_arr[j-2] and s1_arr[i-2] == s2_arr[j-1]:
                d[i,j] = min(d[i,j], d[i-2,j-2] + cost) # Transposition (cost is 1 if s1[i-1]!=s2[j-1], but D-L typically uses cost of 1 for transposition)
                                                        # For OSA, the cost of transposition is 1. We use `cost` which is 0 if chars are same, 1 if different.
                                                        # This is standard Levenshtein with a transposition check.
                                                        # A pure D-L transposition cost is 1.
                                                        # The jellyfish library handles this, so this Numba version is a fallback.
                                                        # For a more accurate D-L, the cost for transposition should be 1.
                                                        # Let's make the transposition cost explicitly 1 for this implementation.
                d[i,j] = min(d[i,j], d[i-2,j-2] + 1)


    return d[len_str1, len_str2]

@lru_cache(maxsize=10000)
def damerau_levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calculates the Damerau-Levenshtein similarity between two strings.
    The score is normalized to be between 0 and 1, where 1 is an exact match.
    Attempts to use jellyfish, falls back to a Numba-jitted OSA version if jellyfish fails.
    """
    logger.debug(f"damerau_levenshtein_similarity called with s1: '{s1}', s2: '{s2}'")
    if not s1 and not s2:
        logger.debug("damerau_levenshtein_similarity returning 1.0 for two empty strings")
        return 1.0
    if not s1 or not s2: # Handles if one is empty but not both
        max_len_for_empty = max(len(s1), len(s2))
        if max_len_for_empty == 0: # Should be caught by previous, but for safety
             logger.debug("damerau_levenshtein_similarity returning 1.0 for two empty strings (edge case)")
             return 1.0
        logger.debug(f"damerau_levenshtein_similarity returning 0.0 for one empty string (len: {max_len_for_empty})")
        return 0.0

    distance = -1
    try:
        distance = jellyfish.damerau_levenshtein_distance(s1, s2)
        logger.debug(f"Used jellyfish.damerau_levenshtein_distance, distance: {distance}")
    except Exception as e: # Catching a more generic exception if jellyfish itself has an issue
        logger.warning(f"jellyfish.damerau_levenshtein_distance failed ('{e}'). Falling back to Numba OSA version.")
        try:
            s1_arr = np.array([ord(c) for c in s1], dtype=np.uint32)
            s2_arr = np.array([ord(c) for c in s2], dtype=np.uint32)
            distance = damerau_levenshtein_distance_numba(s1_arr, s2_arr)
            logger.debug(f"Used Numba OSA version, distance: {distance}")
        except Exception as numba_e:
            logger.error(f"Numba OSA fallback failed for Damerau-Levenshtein: {numba_e}. Returning 0.0 similarity.", exc_info=True)
            return 0.0 # Fallback if Numba version also fails

    if distance == -1: # Should not happen if try-except works
        logger.error("Distance calculation failed for Damerau-Levenshtein. Returning 0.0 similarity.")
        return 0.0

    max_len = max(len(s1), len(s2))
    # max_len will not be 0 here due to earlier checks for empty strings

    similarity = 1.0 - (distance / max_len)
    logger.debug(f"damerau_levenshtein_similarity for '{s1}' vs '{s2}': distance={distance}, max_len={max_len}, similarity={similarity:.4f}")
    return similarity


def monge_elkan_similarity(name1_parts: List[str], name2_parts: List[str], sim_func) -> float:
    """
    Calculates the Monge-Elkan similarity between two lists of name parts (tokens).

    Args:
        name1_parts: A list of strings representing the tokens of the first name.
        name2_parts: A list of strings representing the tokens of the second name.
        sim_func: A secondary similarity function that takes two strings and returns a score between 0 and 1.
                  For example, jaro_winkler_similarity or damerau_levenshtein_similarity.

    Returns:
        The Monge-Elkan similarity score (float).
    """
    logger.debug(f"monge_elkan_similarity called with name1_parts: {name1_parts}, name2_parts: {name2_parts}")

    if not name1_parts: # If name1_parts is empty, score is 0
        logger.debug("Monge-Elkan: name1_parts is empty, returning 0.0")
        return 0.0
    if not name2_parts: # If name2_parts is empty but name1_parts is not, score is also 0
        logger.debug("Monge-Elkan: name2_parts is empty (name1_parts is not), returning 0.0")
        return 0.0

    total_max_similarity = 0.0
    # Filter out empty strings from name1_parts before calculating length for average
    # to avoid division by zero if name1_parts contained only empty strings.
    # However, the problem implies that name1_parts will be a list of actual token strings.
    # The original code already skips empty token1.

    actual_name1_tokens_count = 0
    for token1 in name1_parts:
        if not token1: # Skip empty tokens in the first list
            continue
        actual_name1_tokens_count +=1
        max_sim_for_token1 = 0.0
        for token2 in name2_parts:
            if not token2: # Skip empty tokens in the second list
                continue
            similarity = sim_func(token1, token2)
            if similarity > max_sim_for_token1:
                max_sim_for_token1 = similarity
        total_max_similarity += max_sim_for_token1
        logger.debug(f"Max similarity for token '{token1}' from name1_parts: {max_sim_for_token1:.4f}")

    if actual_name1_tokens_count == 0: # If all tokens in name1_parts were empty
        logger.debug("Monge-Elkan: name1_parts contained only empty strings after filtering, returning 0.0")
        return 0.0

    final_score = total_max_similarity / actual_name1_tokens_count
    logger.debug(f"Monge-Elkan final score: {final_score:.4f}")
    return final_score


def soundex(s: str) -> str:
    """
    Calculate the Soundex code for a string.

    Args:
        s: Input string

    Returns:
        Soundex code (e.g., "W252")
    """
    logger.debug(f"soundex called with s: '{s}'")
    if not s:
        logger.debug("soundex returning '0000' for empty string")
        return "0000"

    # Convert to uppercase
    s = s.upper()

    # Keep the first letter
    first_letter = s[0]

    # Remove non-alphabetic characters
    s = re.sub(r'[^A-Z]', '', s)

    if not s: # If string becomes empty after removing non-alphabetic chars
        # Return a code based on the first letter if it was alphabetic, otherwise 0000
        if 'A' <= first_letter <= 'Z':
            return (first_letter + "000")[:4]
        return "0000"


    # Replace consonants with digits
    s_coded = first_letter # Start with the original first letter

    # Code the rest of the string
    for char_idx in range(1, len(s)):
        char = s[char_idx]
        code = ''
        if char in "BFPV":
            code = "1"
        elif char in "CGJKQSXZ":
            code = "2"
        elif char in "DT":
            code = "3"
        elif char == "L":
            code = "4"
        elif char in "MN":
            code = "5"
        elif char == "R":
            code = "6"
        # Vowels (AEIOU) and H, W, Y are dropped unless they are the first letter (handled by s_coded init)

        # Add code if different from previous and not empty
        if code and (len(s_coded) < 2 or code != s_coded[-1]):
             s_coded += code

    # Remove the first letter's code if it was a vowel, then re-add the letter
    # This step is to handle cases where the first letter is a vowel.
    # The Soundex rule is: Retain the first letter of the name, and drop all other occurrences of a, e, i, o, u, y, h, w.
    # The digit replacement should happen first, then removal of consecutive duplicates, then padding/truncating.

    # Let's refine the Soundex logic based on common descriptions.
    # 1. Retain the first letter of the name.
    # 2. Change all occurrences of the following letters to '0' (zero): A, E, I, O, U, H, W, Y.
    # 3. Change letters to digits as follows:
    #    B, F, P, V → 1
    #    C, G, J, K, Q, S, X, Z → 2
    #    D, T → 3
    #    L → 4
    #    M, N → 5
    #    R → 6
    # 4. Remove all pairs of identical digits if they are adjacent.
    # 5. Remove all zeros from the resulting string.
    # 6. Pad with trailing zeros and return the first four characters (first letter + 3 digits).

    s_upper = s.upper()
    if not s_upper:
        logger.debug("soundex returning '0000' for empty string (after upper)")
        return "0000"

    first_letter = s_upper[0]
    if not 'A' <= first_letter <= 'Z': # Not an alphabet
        logger.debug(f"soundex returning '0000' as first char '{first_letter}' is not alpha")
        return "0000"

    # Rule 2 & 3: Convert letters to codes
    coded_chars = [first_letter]
    soundex_map = {
        'BFPV': '1', 'CGJKQSXZ': '2', 'DT': '3',
        'L': '4', 'MN': '5', 'R': '6'
    }

    for char_code_map, digit in soundex_map.items():
        for char_val in char_code_map:
            s_upper = s_upper.replace(char_val, digit)

    for char in s_upper[1:]: # Start from the second character
        if '1' <= char <= '6':
            # Rule 4: Remove adjacent identical digits (by not adding if same as last)
            if not coded_chars or (coded_chars[-1] != char):
                coded_chars.append(char)
        # Rule 2 (implicitly): Vowels, H, W, Y (and now non-coded consonants) are ignored here

    # Rule 5 is implicitly handled by not adding '0's from rule 2 to coded_chars
    # Rule 6: Pad with zeros and take first four
    coded_string = "".join(coded_chars)
    coded_string = (coded_string + "000")[:4]

    logger.debug(f"soundex for '{s}' (original) -> result: {coded_string}")
    return coded_string


def soundex_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity based on Soundex codes.

    Args:
        s1: First string
        s2: Second string

    Returns:
        1.0 if Soundex codes match, 0.0 otherwise
    """
    logger.debug(f"soundex_similarity called with s1: '{s1}', s2: '{s2}'")
    s1_soundex = soundex(s1)
    s2_soundex = soundex(s2)
    result = 1.0 if s1_soundex == s2_soundex else 0.0
    logger.debug(f"soundex_similarity: soundex1='{s1_soundex}', soundex2='{s2_soundex}', result={result}")
    return result


def jaccard_similarity(s1: str, s2: str) -> float:
    """
    Calculate the Jaccard similarity between two strings based on character n-grams.
    This implementation uses word tokenization, not character n-grams as previously stated.
    For character n-grams, a different tokenization step would be needed.
    """
    logger.debug(f"jaccard_similarity called with s1: '{s1}', s2: '{s2}'")
    # Handle empty strings
    if not s1 and not s2:
        logger.debug("jaccard_similarity returning 1.0 for two empty strings (both empty implies perfect match here)")
        return 1.0 # If both are empty, they are identical
    if not s1 or not s2:
        logger.debug("jaccard_similarity returning 0.0 for one empty string")
        return 0.0  # If one is empty and the other is not, similarity is 0

    # Convert to sets of tokens (words)
    set1 = set(s1.lower().split())
    set2 = set(s2.lower().split())

    if not set1 and not set2: # Both strings were empty or contained only whitespace
        logger.debug("jaccard_similarity returning 1.0 as both token sets are empty")
        return 1.0

    # Calculate Jaccard similarity
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0: # This case should be covered by 'not set1 and not set2'
        return 1.0 if intersection == 0 else 0.0 # Or handle as per requirements for two empty sets

    score = intersection / union
    logger.debug(f"jaccard_similarity for '{s1}' vs '{s2}': intersection={intersection}, union={union}, score={score:.4f}")
    return score


def token_sort_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity between strings after sorting their tokens.
    Useful for names where word order might vary.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score between 0 and 1
    """
    logger.debug(f"token_sort_similarity called with s1: '{s1}', s2: '{s2}'")

    if not s1 and not s2:
        logger.debug("token_sort_similarity: both empty, returning 1.0")
        return 1.0
    if not s1 or not s2:
        logger.debug("token_sort_similarity: one empty, returning 0.0")
        return 0.0

    # Sort tokens
    sorted_s1 = " ".join(sorted(s1.lower().split()))
    sorted_s2 = " ".join(sorted(s2.lower().split()))
    logger.debug(f"token_sort_similarity: sorted_s1='{sorted_s1}', sorted_s2='{sorted_s2}'")

    # Use Jaro-Winkler on the sorted strings
    result = jaro_winkler_similarity(sorted_s1, sorted_s2) # jaro_winkler_similarity already logs
    logger.debug(f"token_sort_similarity for '{s1}' vs '{s2}' (using sorted '{sorted_s1}' vs '{sorted_s2}'): result={result:.4f}")
    return result


def compare_name_components(
    name1_components: Dict[str, str],
    name2_components: Dict[str, str],
    similarity_function
) -> Dict[str, float]:
    """
    Compare individual components of two names using a specified similarity function.

    PERFORMANCE NOTE: This function uses dictionary keys for compatibility.
    For high-performance applications, consider using the optimized version
    with numeric indices in optimized_data_structures.py

    Args:
        name1_components: Components of the first name
        name2_components: Components of the second name
        similarity_function: The function to use for comparing individual string components (e.g., jaro_winkler_similarity).

    Returns:
        Dictionary with similarity scores for each component
    """
    logger.debug(f"compare_name_components called with name1_components: {name1_components}, name2_components: {name2_components}, sim_func: {similarity_function.__name__ if hasattr(similarity_function, '__name__') else 'unknown'}")

    scores = {}
    common_keys = set(name1_components.keys()).union(set(name2_components.keys()))

    for key in ["first_name", "middle_name", "last_name"]: # Ensure these are always present even if None
        val1 = name1_components.get(key, "")
        val2 = name2_components.get(key, "")
        # Ensure sim_func gets strings. If a component is None, treat as empty string.
        scores[key] = similarity_function(val1 if val1 is not None else "", val2 if val2 is not None else "")

    # For full_name_sorted, we use token_sort_similarity which internally uses jaro_winkler_similarity
    # If we want compare_name_components to be fully generic to the similarity_function,
    # this part might need adjustment or to be handled outside.
    # For now, keeping token_sort_similarity as is, which relies on Jaro-Winkler.
    name1_full_for_sort = " ".join(filter(None, [
        name1_components.get("first_name", ""),
        name1_components.get("middle_name", ""),
        name1_components.get("last_name", "")
    ]))
    name2_full_for_sort = " ".join(filter(None, [
        name2_components.get("first_name", ""),
        name2_components.get("middle_name", ""),
        name2_components.get("last_name", "")
    ]))

    scores["full_name_sorted"] = token_sort_similarity(
        name1_full_for_sort,
        name2_full_for_sort
    )

    logger.debug(f"compare_name_components returning scores: {scores}")
    return scores

# Add a GPU-accelerated Levenshtein distance as an example for Numba CUDA
# This function would be called with character arrays (e.g., NumPy arrays of ASCII/UTF-8 bytes)
@cuda.jit(device=True)
def levenshtein_gpu_device(s1_arr, s2_arr, d_arr):
    """
    CUDA device function for Levenshtein distance.
    s1_arr, s2_arr: device arrays of characters (e.g., uint8)
    d_arr: device 2D array for DP table, pre-initialized with row/col 0 values
    This is a simplified example and would need careful indexing and thread management.
    A full, efficient GPU Levenshtein is non-trivial.
    This is more of a placeholder for the concept.
    """
    # Simplified: This would require proper thread indexing (tx, ty, etc.)
    # and careful handling of shared memory if used.
    # For now, let's assume a conceptual single-threaded execution on device for clarity.
    # This is NOT a runnable/efficient GPU kernel as is.

    # A more realistic Numba CUDA approach for string distance might involve:
    # 1. A kernel that processes one pair of strings per thread block or thread.
    # 2. Strings pre-padded or handled with care for varying lengths.
    # 3. Using shared memory for the DP matrix rows if possible.

    # Placeholder logic - actual implementation is complex
    # For example, a thread could compute one cell or one row.
    # This is just to illustrate the @cuda.jit decorator.
    # A real implementation would use threadIdx.x, blockIdx.x, etc.
    # and likely operate on a flattened representation of the DP table
    # or use the two-row optimization.

    # This is a conceptual placeholder and would not run efficiently as-is.
    # A full GPU Levenshtein would be more involved.
    # For demonstration, let's assume this function is called by a kernel
    # that handles the array setup and thread distribution.

    # Example of a single cell calculation (conceptual)
    # i, j = cuda.grid(2) # Example thread indexing
    # if i < d_arr.shape[0] and j < d_arr.shape[1]:
    #     if i == 0:
    #         d_arr[i, j] = j
    #     elif j == 0:
    #         d_arr[i, j] = i
    #     else:
    #         cost = 0 if s1_arr[i-1] == s2_arr[j-1] else 1
    #         d_arr[i,j] = min(d_arr[i-1, j] + 1, d_arr[i, j-1] + 1, d_arr[i-1, j-1] + cost)
    pass # Placeholder for actual GPU kernel logic


def levenshtein_similarity_gpu_host(s1_str: str, s2_str: str) -> float:
    """
    Host function to demonstrate calling a conceptual GPU Levenshtein.
    This is a placeholder to show where GPU acceleration could be applied.
    """
    logger.debug(f"levenshtein_similarity_gpu_host called for '{s1_str}' and '{s2_str}'")
    if not CUDA_AVAILABLE:
        logger.warning("CUDA not available for levenshtein_similarity_gpu_host. Falling back to CPU (jellyfish).")
        # Fallback to a CPU version (e.g., jellyfish or a Numba CPU JIT version)
        try:
            distance = jellyfish.levenshtein_distance(s1_str, s2_str)
        except Exception as e:
            logger.error(f"Fallback jellyfish.levenshtein_distance failed: {e}")
            return 0.0 # Or handle error appropriately
        max_len = max(len(s1_str), len(s2_str))
        if max_len == 0: return 1.0 if not s1_str and not s2_str else 0.0
        return 1.0 - (distance / max_len)

    # --- Conceptual GPU execution path ---
    # 1. Convert strings to NumPy arrays of character codes
    s1_arr_host = np.array([ord(c) for c in s1_str], dtype=np.uint8) # Or uint32 for full Unicode
    s2_arr_host = np.array([ord(c) for c in s2_str], dtype=np.uint8)

    # 2. Allocate memory on GPU and copy data
    s1_gpu = cuda.to_device(s1_arr_host)
    s2_gpu = cuda.to_device(s2_arr_host)

    # DP table for Levenshtein, size (len(s1)+1) x (len(s2)+1)
    # For a real GPU kernel, this would be managed carefully.
    # This is a simplified placeholder.
    # For actual pairwise comparisons, you'd likely launch many kernels.
    # For this example, we'll simulate calculating one distance.

    # If we were to implement a full Levenshtein on GPU:
    # dp_gpu = cuda.device_array((len(s1_arr_host) + 1, len(s2_arr_host) + 1), dtype=np.int32)
    # Initialize first row and col of dp_gpu
    # Define kernel launch configuration (blocks, threads)
    # Call levenshtein_gpu_device[blocks_per_grid, threads_per_block](s1_gpu, s2_gpu, dp_gpu)
    # distance = dp_gpu[len(s1_arr_host), len(s2_arr_host)].copy_to_host()

    # For this conceptual example, let's just say it would compute the distance.
    # Since implementing a full, efficient GPU Levenshtein here is too complex,
    # we'll acknowledge the concept and for actual execution,
    # it would fall back or use a CPU version for now.
    logger.info("Conceptual GPU path for Levenshtein. Actual GPU kernel not fully implemented here for brevity.")
    logger.warning("levenshtein_similarity_gpu_host is conceptual and falls back to CPU jellyfish.")

    try:
        distance = jellyfish.levenshtein_distance(s1_str, s2_str)
    except Exception as e:
        logger.error(f"Fallback jellyfish.levenshtein_distance failed in GPU path: {e}")
        return 0.0
    max_len = max(len(s1_str), len(s2_str))
    if max_len == 0: return 1.0 if not s1_str and not s2_str else 0.0
    return 1.0 - (distance / max_len)
