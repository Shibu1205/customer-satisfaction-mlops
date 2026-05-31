import logging
from typing import Annotated, Optional, Tuple

import pandas as pd
from zenml import step

from src.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreprocessStrategy,
)





@step(enable_cache=False)
def clean_df(
    df: Optional[pd.DataFrame] = None,
    data_path: Optional[str] = None,
) -> Tuple[

    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:

    try:
        if df is None:
            if data_path is None:
                raise ValueError("Either df or data_path must be provided.")
            logging.info(f"Reading data from {data_path}")
            df = pd.read_csv(data_path)

        preprocess_strategy = DataPreprocessStrategy()

        data_cleaning = DataCleaning(
            data=df,
            strategy=preprocess_strategy,
        )

        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()

        data_cleaning = DataCleaning(
            data=processed_data,
            strategy=divide_strategy,
        )

        X_train, X_test, y_train, y_test = (
            data_cleaning.handle_data()
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
        )

    except Exception as e:

        logging.error(f"Error in clean_df step: {e}")

        raise e
    
    
