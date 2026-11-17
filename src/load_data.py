
from pathlib import Path
import json
import logging

import pandas as pd


# Find the project and raw-data folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "DATA" / "raw"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


def load_csv(filename):
    """Load a CSV file safely."""
    file_path = DATA_DIR / filename

    try:
        df = pd.read_csv(file_path)
        logging.info(
            "Loaded %s: %s rows, %s columns",
            filename,
            df.shape[0],
            df.shape[1]
        )
        return df

    except Exception as error:
        logging.error("Could not load %s: %s", filename, error)
        return None


def load_json(filename):
    """Load a JSON file safely."""
    file_path = DATA_DIR / filename

    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        df = pd.json_normalize(data)

        logging.info(
            "Loaded %s: %s rows, %s columns",
            filename,
            df.shape[0],
            df.shape[1]
        )
        return df

    except Exception as error:
        logging.error("Could not load %s: %s", filename, error)
        return None


def load_all_datasets():
    """Load all eight NorthStar datasets."""

    datasets = {
        "transactions": load_csv("transactions.csv"),
        "customers": load_csv("customers.csv"),
        "products": load_csv("products.csv"),
        "stores": load_csv("stores.csv"),
        "returns": load_csv("returns.csv"),
        "inventory": load_csv("inventory.csv"),
        "campaigns": load_json("campaigns.json"),
        "store_targets": load_csv("store_targets_legacy.csv"),
    }

    successful = sum(
        df is not None for df in datasets.values()
    )

    logging.info(
        "Ingestion summary: %s of %s datasets loaded.",
        successful,
        len(datasets)
    )

    return datasets


if __name__ == "__main__":
    datasets = load_all_datasets()

    print("\n--- INGESTION SUMMARY ---")

    for name, df in datasets.items():
        if df is not None:
            print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print(f"{name}: FAILED")
