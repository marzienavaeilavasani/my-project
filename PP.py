"""
All-in-One Python Program for Training, Ideal Function Selection, Test Mapping, and Visualization
Author: Your Name
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from bokeh.plotting import figure, show
from bokeh.models import Legend
import unittest

# -------------------------------
# Exceptions
# -------------------------------
class DataLoadError(Exception):
    """Raised when CSV or database loading fails."""
    pass

class MappingError(Exception):
    """Raised when a test point cannot be mapped to any ideal function."""
    pass

# -------------------------------
# Database Models
# -------------------------------
Base = declarative_base()

class TrainingData(Base):
    __tablename__ = 'training_data'
    X = Column(Float, primary_key=True)
    Y1 = Column(Float)
    Y2 = Column(Float)
    Y3 = Column(Float)
    Y4 = Column(Float)

class IdealFunctions(Base):
    __tablename__ = 'ideal_functions'
    X = Column(Float, primary_key=True)
    # Create 50 ideal function columns dynamically
    for i in range(1, 51):
        vars()[f'Y{i}'] = Column(Float)

class TestResults(Base):
    __tablename__ = 'test_results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    X = Column(Float)
    Y = Column(Float)
    Delta_Y = Column(Float)
    IdealFuncNo = Column(Integer)

def get_session(db_name="functions.db"):
    engine = create_engine(f"sqlite:///{db_name}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()

# -------------------------------
# CSV Loader Classes (with Inheritance)
# -------------------------------
class CSVLoader:
    """Base class for loading CSV files."""
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        try:
            df = pd.read_csv(self.filepath)
            return df
        except Exception as e:
            raise DataLoadError(f"Failed to load {self.filepath}: {e}")

class TrainingCSVLoader(CSVLoader):
    """Load training CSV."""
    pass

class IdealCSVLoader(CSVLoader):
    """Load ideal function CSV."""
    pass

class TestCSVLoader(CSVLoader):
    """Load test CSV."""
    pass

# -------------------------------
# Ideal Function Selector
# -------------------------------
class IdealFunctionSelector:
    """Selects best matching ideal functions using least squares."""
    def __init__(self, training_df, ideal_df):
        self.training_df = training_df
        self.ideal_df = ideal_df
        self.selected_funcs = []

    def select_best_functions(self, top_n=4):
        deviations = {}
        for col in self.ideal_df.columns[1:]:
            total_dev = 0
            for t_col in self.training_df.columns[1:]:
                diff = self.training_df[t_col].values - self.ideal_df[col].values
                total_dev += np.sum(diff**2)
            deviations[col] = total_dev
        self.selected_funcs = sorted(deviations, key=deviations.get)[:top_n]
        return self.selected_funcs

    def map_test_point(self, x, y, max_devs):
        """Map single test point to one of the selected ideal functions."""
        best_func = None
        min_dev = float('inf')
        for func in self.selected_funcs:
            ideal_y = self.ideal_df.loc[self.ideal_df['X']==x, func].values[0]
            dev = abs(y - ideal_y)
            if dev <= max_devs[func] * np.sqrt(2) and dev < min_dev:
                min_dev = dev
                best_func = func
        if best_func is None:
            raise MappingError(f"Test point ({x}, {y}) could not be mapped")
        return best_func, min_dev

# -------------------------------
# Test Data Mapper
# -------------------------------
class TestDataMapper:
    """Map test CSV data to ideal functions and save results in database."""
    def __init__(self, test_csv, selector, max_devs):
        self.test_df = pd.read_csv(test_csv)
        self.selector = selector
        self.max_devs = max_devs
        self.session = get_session()

    def map_and_save(self):
        for _, row in self.test_df.iterrows():
            try:
                func, delta = self.selector.map_test_point(row['X'], row['Y'], self.max_devs)
                result = TestResults(X=row['X'], Y=row['Y'], Delta_Y=delta, IdealFuncNo=int(func[1:]))
                self.session.add(result)
            except MappingError as e:
                print(e)
        self.session.commit()

# -------------------------------
# Visualization
# -------------------------------
def plot_functions(training_df, ideal_df, selected_funcs, test_results):
    p = figure(title="Functions and Test Points", x_axis_label='X', y_axis_label='Y')
    
    # Training functions
    for col in training_df.columns[1:]:
        p.line(training_df['X'], training_df[col], line_width=2, color='blue', legend_label=col)
    
    # Selected ideal functions
    for col in selected_funcs:
        p.line(ideal_df['X'], ideal_df[col], line_width=2, color='green', legend_label=col)
    
    # Test points
    if not test_results.empty:
        p.circle(test_results['X'], test_results['Y'], size=5, color='red', legend_label='Test Data')
    
    show(p)

# -------------------------------
# Main Execution
# -------------------------------
def main():
    # Load training and ideal datasets
    training_df = TrainingCSVLoader("training.csv").load()
    ideal_df = IdealCSVLoader("ideal_functions.csv").load()

    # Select best matching ideal functions
    selector = IdealFunctionSelector(training_df, ideal_df)
    best_funcs = selector.select_best_functions(top_n=4)
    print("Selected Ideal Functions:", best_funcs)

    # Compute max deviations from training for mapping criterion
    max_devs = {}
    for func in best_funcs:
        deviations = [abs(training_df[col]-ideal_df[func]) for col in training_df.columns[1:]]
        max_devs[func] = max([max(d) for d in deviations])

    # Map test data
    mapper = TestDataMapper("test_data.csv", selector, max_devs)
    mapper.map_and_save()

    # Fetch test results
    session = get_session()
    test_results_query = session.query(TestResults).all()
    test_results_df = pd.DataFrame([{'X':r.X,'Y':r.Y,'Delta_Y':r.Delta_Y,'IdealFuncNo':r.IdealFuncNo} for r in test_results_query])

    # Visualize
    plot_functions(training_df, ideal_df, best_funcs, test_results_df)

# -------------------------------
# Unit Tests
# -------------------------------
class TestIdealFunctionSelector(unittest.TestCase):
    def setUp(self):
        self.training_df = pd.DataFrame({'X':[1,2], 'Y1':[1,2], 'Y2':[2,3]})
        self.ideal_df = pd.DataFrame({'X':[1,2], 'Y1':[1.1,1.9], 'Y2':[2.1,3.1]})
        self.selector = IdealFunctionSelector(self.training_df, self.ideal_df)

    def test_select_best_functions(self):
        best = self.selector.select_best_functions(top_n=1)
        self.assertIn('Y1', best)

# -------------------------------
# Run main
# -------------------------------
if __name__ == "__main__":
    main()
    # Run unit tests
    unittest.main(argv=[''], exit=False)
