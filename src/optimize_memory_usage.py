import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def optimize_memory_usage(
    df: pd.DataFrame,
    category_threshold: float = 0.5,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Optimizes the memory footprint of a Pandas DataFrame by downcasting numeric 
    columns and converting low-cardinality string/object columns to categories.

    Args:
        df (pd.DataFrame): The input pandas DataFrame to be optimized.
        category_threshold (float, optional): The ratio of unique values to total rows 
            below which an object column will be converted to categorical. 
            Must be between 0.0 and 1.0. Defaults to 0.5.
        inplace (bool, optional): If True, modifies the DataFrame in place. 
            If False, returns a new DataFrame. Defaults to False.

    Returns:
        pd.DataFrame: A memory-optimized DataFrame.

    Raises:
        ValueError: If `df` is not a DataFrame, or if `category_threshold` is 
            not between 0.0 and 1.0.
        Exception: If an unexpected error occurs during processing.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = {
        ...     'int_col': np.random.randint(0, 100, size=1000),
        ...     'float_col': np.random.random(1000),
        ...     'cat_col': ['A', 'B', 'C', 'D'] * 250
        ... }
        >>> df = pd.DataFrame(data)
        >>> df['int_col'] = df['int_col'].astype('int64')
        >>> df['float_col'] = df['float_col'].astype('float64')
        >>> optimized_df = optimize_memory_usage(df, category_threshold=0.3)
    """

    # 1. Fail fast input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected a pandas DataFrame, but got {type(df)}")
    
    if not (0.0 <= category_threshold <= 1.0):
        raise ValueError("category_threshold must be a float between 0.0 and 1.0")

    # 2. Inplace logic
    if not inplace:
        df = df.copy()

    logger.debug("Starting processing...")

    try:
        # 3. Core logic
        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        logger.debug(f"Initial memory usage: {start_mem:.2f} MB")

        for col in df.columns:
            # Handle Numeric Types
            if pd.api.types.is_numeric_dtype(df[col]):
                if pd.api.types.is_integer_dtype(df[col]):
                    # Downcast unsigned integers if min >= 0, else signed integers
                    if df[col].min() >= 0:
                        df[col] = pd.to_numeric(df[col], downcast='unsigned')
                    else:
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                elif pd.api.types.is_float_dtype(df[col]):
                    # Downcast floats to smallest possible representation
                    df[col] = pd.to_numeric(df[col], downcast='float')
            
            # Handle Object/String Types
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                num_unique_values = df[col].nunique(dropna=False)
                num_total_values = len(df[col])
                
                if num_total_values > 0:
                    cardinality_ratio = num_unique_values / num_total_values
                    if cardinality_ratio < category_threshold:
                        df[col] = df[col].astype('category')

        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        reduction_percentage = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
        
        logger.debug(f"Final memory usage: {end_mem:.2f} MB (Decreased by {reduction_percentage:.1f}%)")
        logger.info("Processing complete.")

    except Exception as e:
        logger.exception("Unexpected error occurred.")
        raise e

    return df