import logging
import random

import kagglehub
import numpy as np
import pandas as pd

from src.config import seed

logging.basicConfig(level=logging.INFO)


def data_loader() -> pd.DataFrame:
    """Download the Telco Customer Churn dataset from Kaggle and return it as a DataFrame.

    Returns:
        pd.DataFrame: Raw dataset with 21 columns and ~7043 rows.
    """

    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    np.random.seed(seed)
    random.seed(seed)

    df = pd.read_csv(path + "/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    logging.info(df.info())
    return df.copy()
