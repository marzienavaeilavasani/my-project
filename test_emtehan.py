import pytest
import pandas as pd
import numpy as np
from io import StringIO
from emtehan import (
    DataLoader,
    TrainingDataLoader,
    DatabaseHandler,
    ModelSelector,
    TestDataMapper,
    DataError,
    Visualizer
)

# ------------------ DataLoader Tests ------------------
def test_load_csv_success(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("X,Y\n1,2\n3,4")
    loader = DataLoader(str(csv_file))
    df = loader.load()
    assert df.shape == (2, 2)
    assert list(df.columns) == ["X", "Y"]

def test_load_csv_file_not_found():
    loader = DataLoader("nonexistent.csv")
    with pytest.raises(DataError, match="File not found"):
        loader.load()

def test_load_csv_empty_file(tmp_path):
    csv_file = tmp_path / "empty.csv"
    csv_file.write_text("")
    loader = DataLoader(str(csv_file))
    with pytest.raises(DataError, match="File empty"):
        loader.load()

# ------------------ DatabaseHandler Tests ------------------
def test_save_dataframe_in_memory():
    db = DatabaseHandler("sqlite:///:memory:")
    df = pd.DataFrame({"X": [1,2], "Y": [3,4]})
    # Should not raise
    db.save_dataframe(df, "test_table")

# ------------------ ModelSelector Tests ------------------
def test_find_best_fits():
    train_df = pd.DataFrame({"X": [1,2], "A": [1,2], "B": [2,3]})
    ideal_df = pd.DataFrame({"X": [1,2], "I1": [1,2], "I2": [2,3]})
    selector = ModelSelector(train_df, ideal_df)
    mapping = selector.find_best_fits()
    assert mapping == {"A": "I1", "B": "I2"}

# ------------------ TestDataMapper Tests ------------------
def test_map_points_within_deviation():
    test_df = pd.DataFrame({"X": [1], "Y": [1.1]})
    ideal_df = pd.DataFrame({"X": [1], "I1": [1.0]})
    mapping = {"A": "I1"}
    max_dev = {"A": 0.2}
    mapper = TestDataMapper(test_df, ideal_df, mapping, max_dev)
    mapped = mapper.map_points()
    assert mapped.shape[0] == 1
    assert mapped.iloc[0]["DeltaY"] == pytest.approx(0.1)

def test_map_points_out_of_deviation():
    test_df = pd.DataFrame({"X": [1], "Y": [1.5]})
    ideal_df = pd.DataFrame({"X": [1], "I1": [1.0]})
    mapping = {"A": "I1"}
    max_dev = {"A": 0.2}
    mapper = TestDataMapper(test_df, ideal_df, mapping, max_dev)
    mapped = mapper.map_points()
    assert mapped.shape[0] == 0

# ------------------ Visualizer Tests ------------------
def test_visualizer_runs():
    training_df = pd.DataFrame({"X": [1,2], "A": [1,2]})
    ideal_df = pd.DataFrame({"X": [1,2], "I1": [1,2]})
    test_df = pd.DataFrame({"X": [1], "Y": [1]})
    vis = Visualizer()
    # Should not raise
    vis.plot(training_df, ideal_df, test_df)
