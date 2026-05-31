import logging
import pandas as pd

from zenml import step


class IngestData:

    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path)


@step(enable_cache=False)
def ingest_df(data_path: str) -> pd.DataFrame:

    try:
        logging.info(f"Ingesting data from {data_path}")

        ingest_data = IngestData(data_path)

        df = ingest_data.get_data()

        print(df.head())

        return df

    except Exception as e:

        logging.error(f"Error in ingesting data: {e}")

        raise e
    
    