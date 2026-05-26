import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def standardize_column_names(
    df: pd.DataFrame,
    lowercase: bool = True,
    replace_spaces_with: str = "_",
    remove_special_chars: bool = True,
    strip_accents: bool = True,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Standardizes the column names of a pandas DataFrame with Unicode support.

    This function cleans column names by stripping leading/trailing whitespace,
    optionally converting them to lowercase, and replacing spaces. It safely 
    handles international characters (like Kanjis or Arabic) and can optionally 
    flatten Latin diacritics (accents).

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        lowercase (bool, optional): Whether to convert column names to lowercase. Defaults to True.
        replace_spaces_with (str, optional): The string to replace spaces with. Defaults to "_".
        remove_special_chars (bool, optional): Whether to remove punctuation and symbols. 
            Uses Unicode-aware regex to retain letters from any language, digits, and underscores. Defaults to True.
        strip_accents (bool, optional): Whether to remove diacritics/accents from Latin characters 
            (e.g., 'á' becomes 'a'). Defaults to True.
        inplace (bool, optional): If True, modifies the DataFrame in place. Defaults to False.

    Returns:
        pd.DataFrame: The DataFrame with standardized column names.

    Raises:
        TypeError: If the input 'df' is not a pandas DataFrame.
        Exception: If an unexpected error occurs during processing.

    Example:
        >>> import pandas as pd
        >>> data = {' Café! ': [1], 'Niño_Age': [2], '売上 (Sales)': [3], 'الإيرادات': [4]}
        >>> df = pd.DataFrame(data)
        >>> df = standardize_column_names(df, strip_accents=True)
        >>> df.columns.tolist()
        ['cafe', 'nino_age', '売上_sales', 'الإيرادات']
    """
    
    # 1. Fail fast input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but got {type(df).__name__}.")

    # 2. Inplace logic
    if not inplace:
        df = df.copy()

    logger.debug("Starting column name standardization...")

    try:
        # 3. Core logic
        new_columns = []
        for col in df.columns:
            col_str = str(col).strip()
            
            # Remove diacritics (accents) but keep base characters
            if strip_accents:
                # NFD decomposes characters into base + combining characters (accents)
                # We then filter out the combining characters ('Mn' category)
                col_str = ''.join(
                    c for c in unicodedata.normalize('NFD', col_str)
                    if unicodedata.category(c) != 'Mn'
                )
            
            if lowercase:
                col_str = col_str.lower()
            
            col_str = col_str.replace(" ", replace_spaces_with)
            
            if remove_special_chars:
                # \w matches any Unicode word character (letters from any alphabet, digits, and underscores)
                # [^\w] will target and remove only non-word characters (like punctuation, symbols)
                col_str = re.sub(r'[^\w]', '', col_str)
                
            new_columns.append(col_str)
            
        df.columns = new_columns
        
        logger.info("Processing complete. Column names have been standardized.")

    except Exception as e:
        logger.exception("Unexpected error occurred while standardizing column names.")
        raise e

    return df