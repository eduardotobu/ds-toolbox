# 🧰 ds-toolbox
A comprehensive toolkit for data science workflows. Features production-ready Python utility functions, standardized project templates, and instructional notebooks for end-to-end analysis.

## 📁 Repository Structure
This repository acts as a central hub for standardizing data science work. It is organized into three main components: tools, blueprints, and instructions.

```Plaintext
ds-toolbox/
├── templates/             # Starter directories to clone for new projects
│   └── basic-ds-project/  # An empty scaffold with a standard README and requirements
├── notebooks/             # Instructional workflows and analysis examples
│   ├── 01-eda-workflow.ipynb
│   └── 02-feature-engineering-guide.ipynb
├── src/                   # Production-ready Python utility snippets
│   ├── data/              # Data ingestion and cleaning
│   ├── features/          # Feature engineering and transformations
│   ├── models/            # Training, prediction, and evaluation wrappers
│   └── utils/             # Generic helpers (logging, config parsers)
├── tests/                 # Unit tests for the src/ snippets
├── README.md              
└── requirements.txt       # Core dependencies for the toolkit
```
## 🧩 Toolkit Components
`src/` **(The Tools)**: A modular library of pure, well-documented Python functions. These replace messy, repetitive code blocks in your notebooks.

`templates/` **(The Blueprints)**: Ready-to-copy directory structures for new data science projects, ensuring every new analysis starts with the same robust foundation.

`notebooks/` **(The Instructions)**: Curated, highly-commented Jupyter notebooks demonstrating best practices for common workflows (e.g., Exploratory Data Analysis, Model Evaluation).

## 🎯 Core Best Practices
All code contributed to the `src/` directory must adhere to the following principles:

- **No `print()` statements**: Always use the `logging` module. Print statements get lost in production; logs can be routed and filtered by severity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

- **Pure Functions by Default**: Avoid side effects. Data should flow in as an input and out as a return value. Do not modify global variables. If memory management is a concern, require an explicit `inplace=True` argument.

- **Google-Style Docstrings**: Standardized docstrings allow for automated documentation generation and provide clear instructions.

- **Fail Fast**: Validate inputs and raise explicit errors (e.g., `KeyError`, `ValueError`, `TypeError`) at the top of the function before starting heavy computations.

- **Type Hinting**: Use the `typing` module to clearly define expected input and output data types.

## 🏗️ The Function Template
Below is the standard template that must be used for all new utility functions.

```Python
import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def template_data_function(
    df: pd.DataFrame,
    target_column: str,
    optional_param: Optional[float] = 1.0,
    inplace: bool = False
) -> pd.DataFrame:
    """
    [One-line summary: Action verb + what it does to what object].

    [Optional extended description: Explain the "why" and "how" if the logic 
    is complex. Mention any underlying math, assumptions, or specific business rules.]

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to process.
        target_column (str): The name of the column to apply the transformation to.
        optional_param (float, optional): A parameter that tweaks the logic. Defaults to 1.0.
        inplace (bool, optional): If True, modifies original DataFrame in memory. Defaults to False.

    Returns:
        pd.DataFrame: The transformed DataFrame.

    Raises:
        KeyError: If `target_column` is not found in the DataFrame.
        ValueError: If `optional_param` is outside the acceptable range.
        TypeError: If the target column is not of the expected data type.
        
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3]})
        >>> template_data_function(df, target_column='A', optional_param=2.0)
           A  new_A
        0  1    2.0
        1  2    4.0
        2  3    6.0
    """
    
    # 1. Input Validation & Error Handling (Fail Fast)
    if target_column not in df.columns:
        logger.error(f"Column '{target_column}' missing from DataFrame.")
        raise KeyError(f"Column '{target_column}' not found in DataFrame.")

    if optional_param < 0:
        logger.error(f"optional_param must be >= 0. Received: {optional_param}")
        raise ValueError("optional_param cannot be negative.")

    # 2. Handle 'inplace' logic cleanly
    if not inplace:
        df = df.copy()

    logger.debug(f"Starting processing on column: {target_column}")

    try:
        # ==========================================
        # 3. CORE LOGIC GOES HERE
        df[f"new_{target_column}"] = df[target_column] * optional_param
        # ==========================================

        logger.info(
            f"Successfully processed '{target_column}'. "
            f"Created new feature 'new_{target_column}'."
        )

    except Exception as e:
        logger.exception(f"Unexpected error during processing of '{target_column}'.")
        raise e

    return df
```

## 🤖 LLM Generation Prompt
This repository structure is highly LLM-friendly. To rapidly generate new functions using an AI, copy and paste the following prompt. It forces the model to adhere strictly to our repository's standards.

```Plaintext
You are an expert Data Science Python Developer. 
Your task is to write a Python function based on my request.

CRITICAL RULES:
1. You MUST use the exact structure, error handling, and logging style provided in the TEMPLATE below.
2. Do not use print() statements; only use the logger.
3. Include a complete Google-style docstring with an Example section.
4. Keep the function pure (do not modify global variables unless inplace=True is explicitly requested).
5. Always use Type Hints.

--- START TEMPLATE ---
import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def template_function(
    df: pd.DataFrame,
    # Add your args here
    inplace: bool = False
) -> pd.DataFrame:
    """
    [Google Style Docstring]
    """
    
    # 1. Fail fast input validation
    # 2. Inplace logic
    if not inplace:
        df = df.copy()

    logger.debug("Starting processing...")

    try:
        # 3. Core logic
        pass
        
        logger.info("Processing complete.")

    except Exception as e:
        logger.exception("Unexpected error occurred.")
        raise e

    return df

--- END TEMPLATE ---

MY REQUEST:
[Describe the function you want to build here. E.g., "Write a function called `clip_extreme_values` that takes a DataFrame and a list of numeric columns, and clips the values at the 1st and 99th percentiles."]
```
