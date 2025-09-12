import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from bokeh.plotting import figure, show
import math

# ----------------- Exceptions -----------------
class DataError(Exception):
    """Custom exception for data handling errors"""
    pass


# ----------------- Base Class -----------------
class DataLoader:
    """Base class for loading CSV files into pandas DataFrames"""
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.filepath)
        except FileNotFoundError:
            raise DataError(f"File not found: {self.filepath}")
        except pd.errors.EmptyDataError:
            raise DataError(f"File empty: {self.filepath}")


# ----------------- Inherited Class -----------------
class TrainingDataLoader(DataLoader):
    """Loads training datasets"""
    pass


# ----------------- Database Handler -----------------
class DatabaseHandler:
    """Handles saving and loading data to SQLite"""
    def __init__(self, db_name="project.db"):
        self.engine = create_engine(f"sqlite:///{db_name}")

    def save_dataframe(self, df: pd.DataFrame, table_name: str):
        try:
            df.to_sql(table_name, self.engine, if_exists="replace", index=False)
        except SQLAlchemyError as e:
            raise DataError(f"Database error: {e}")


# ----------------- Model Selector -----------------
class ModelSelector:
    """Selects best fit ideal functions using least squares"""
    def __init__(self, training_df: pd.DataFrame, ideal_df: pd.DataFrame):
        self.training_df = training_df
        self.ideal_df = ideal_df

    def find_best_fits(self) -> dict:
        """Return dict of 4 best matching ideal functions"""
        best_funcs = {}
        for col in self.training_df.columns[1:]:  # skip X
            y_train = self.training_df[col].values
            min_error = float("inf")
            best_func = None
            for ideal_col in self.ideal_df.columns[1:]:
                y_ideal = self.ideal_df[ideal_col].values
                error = np.sum((y_train - y_ideal)**2)
                if error < min_error:
                    min_error = error
                    best_func = ideal_col
            best_funcs[col] = best_func
        return best_funcs


# ----------------- Test Data Mapper -----------------
class TestDataMapper:
    """Maps test data to chosen functions if within allowed deviation"""
    def __init__(self, test_df: pd.DataFrame, ideal_df: pd.DataFrame, mapping: dict, max_dev: dict):
        self.test_df = test_df
        self.ideal_df = ideal_df
        self.mapping = mapping
        self.max_dev = max_dev

    def map_points(self) -> pd.DataFrame:
        results = []
        for _, row in self.test_df.iterrows():
            x, y = row["X"], row["Y"]
            for train_col, ideal_col in self.mapping.items():
                y_ideal = self.ideal_df.loc[self.ideal_df["X"] == x, ideal_col].values[0]
                deviation = abs(y - y_ideal)
                if deviation <= self.max_dev[train_col] * math.sqrt(2):
                    results.append([x, y, deviation, ideal_col])
        return pd.DataFrame(results, columns=["X", "Y", "DeltaY", "IdealFunction"])


# ----------------- Visualizer -----------------
class Visualizer:
    """Visualizes training, ideal, and test mappings"""
    def plot(self, training_df, ideal_df, test_df):
        p = figure(title="Training vs Ideal vs Test Data", x_axis_label="X", y_axis_label="Y")
        # Training data
        for col in training_df.columns[1:]:
            p.line(training_df["X"], training_df[col], legend_label=f"Train {col}")
        # Test data
        p.scatter(test_df["X"], test_df["Y"], legend_label="Test Data", size=6, color="red")
        show(p)


# ----------------- Main -----------------
def main():
    # Load data
    train = TrainingDataLoader(r"training.csv").load()
    ideal = DataLoader(r"ideal.csv").load()
    test = DataLoader(r"test.csv").load()

    # Save to DB
    db = DatabaseHandler()
    db.save_dataframe(train, "training_data")
    db.save_dataframe(ideal, "ideal_functions")
    db.save_dataframe(test, "test_data")

    # Select best fits
    selector = ModelSelector(train, ideal)
    best_mapping = selector.find_best_fits()
    print("Best mapping:", best_mapping)

    # Compute max deviations
    max_dev = {}
    for train_col, ideal_col in best_mapping.items():
        max_dev[train_col] = np.max(abs(train[train_col].values - ideal[ideal_col].values))

    # Map test data
    mapper = TestDataMapper(test, ideal, best_mapping, max_dev)
    mapped = mapper.map_points()
    db.save_dataframe(mapped, "mapped_test_data")

    # Visualize
    vis = Visualizer()
    vis.plot(train, ideal, test)


if __name__ == "__main__":
    main()
