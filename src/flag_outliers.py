import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def flag_outliers(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "iqr",
    threshold: Optional[float] = None,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Flags outliers in specified numeric columns using Z-score or Interquartile Range (IQR).
    
    This function creates new boolean columns named `{column_name}_outlier` which are 
    True if the row is deemed an outlier based on the selected statistical method.

    Args:
        df (pd.DataFrame): The pandas DataFrame to process.
        columns (List[str]): A list of numeric column names to check for outliers.
        method (str, optional): The method to use for outlier detection. 
            Must be either 'iqr' or 'zscore'. Defaults to 'iqr'.
        threshold (Optional[float], optional): The threshold to determine an outlier. 
            If None, defaults to 1.5 for 'iqr' and 3.0 for 'zscore'. Defaults to None.
        inplace (bool, optional): If True, modifies the DataFrame in place. 
            If False, returns a new copy. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame with the new boolean outlier flag columns.

    Raises:
        TypeError: If `df` is not a pandas DataFrame.
        ValueError: If `columns` is empty, contains missing columns, or if `method` is invalid.

    Example:
        >>> import pandas as pd
        >>> data = {'A': [10, 12, 12, 13, 12, 100], 'B': [1, 2, 3, 4, 5, 6]}
        >>> df = pd.DataFrame(data)
        >>> df_flagged = flag_outliers(df, columns=['A', 'B'], method='iqr')
        >>> df_flagged['A_outlier'].tolist()
        [False, False, False, False, False, True]
    """
    
    # 1. Fail fast input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    
    if not columns:
        raise ValueError("The 'columns' list cannot be empty.")
        
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
    valid_methods = {"iqr", "zscore"}
    method = method.lower()
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}, got '{method}'.")

    # Set appropriate default thresholds based on the method
    if threshold is None:
        threshold = 3.0 if method == "zscore" else 1.5

    # 2. Inplace logic
    if not inplace:
        df = df.copy()

    logger.debug(f"Starting outlier flagging using {method.upper()} with threshold {threshold}...")

    try:
        # 3. Core logic
        for col in columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning("Column '%s' is not numeric. Skipping outlier detection.", col)
                continue
                
            flag_col_name = f"{col}_outlier"
            
            if method == "iqr":
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - (threshold * iqr)
                upper_bound = q3 + (threshold * iqr)
                
                df[flag_col_name] = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == "zscore":
                mean = df[col].mean()
                std = df[col].std(ddof=0)
                
                if std == 0:
                    # If standard deviation is 0, there is no variance, hence no outliers
                    df[flag_col_name] = False
                else:
                    z_scores = (df[col] - mean) / std
                    df[flag_col_name] = z_scores.abs() > threshold

        logger.info("Processing complete. Outlier flags successfully generated.")

    except Exception as e:
        logger.exception("Unexpected error occurred.")
        raise e

    return df