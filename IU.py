import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_csv_files(folder):
    """Load all CSV files in a folder into a dictionary {filename: dataframe}"""
    csv_files = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]
    data = {}
    for file in csv_files:
        df = pd.read_csv(os.path.join(folder, file))
        data[file] = df
    return data

def compute_squared_error(train_df, ideal_df):
    """Compute sum of squared deviations for aligned X values"""
    # Interpolate ideal Y at train X values
    ideal_y_interp = np.interp(train_df['X'], ideal_df['X'], ideal_df['Y'])
    error = np.sum((train_df['Y'] - ideal_y_interp) ** 2)
    max_dev = np.max(np.abs(train_df['Y'] - ideal_y_interp))
    return error, max_dev

def select_best_ideals(training_data, ideal_data):
    """Select one ideal function per training dataset that minimizes squared error"""
    selected = {}
    max_devs = {}
    for t_name, t_df in training_data.items():
        best_error = float('inf')
        best_ideal = None
        best_max_dev = None
        for i_name, i_df in ideal_data.items():
            error, max_dev = compute_squared_error(t_df, i_df)
            if error < best_error:
                best_error = error
                best_ideal = i_name
                best_max_dev = max_dev
        selected[t_name] = best_ideal
        max_devs[t_name] = best_max_dev
        print(f"Training '{t_name}' best ideal: '{best_ideal}' with squared error {best_error:.2f}")
    return selected, max_devs

def map_test_points(test_df, selected_ideals, ideal_data, max_devs):
    """Map test points to the selected ideal functions using deviation criterion"""
    mapped_points = []
    for idx, row in test_df.iterrows():
        x, y = row['X'], row['Y']
        for train_name, ideal_name in selected_ideals.items():
            ideal_df = ideal_data[ideal_name]
            ideal_y = np.interp(x, ideal_df['X'], ideal_df['Y'])
            deviation = abs(y - ideal_y)
            if deviation <= max_devs[train_name] * np.sqrt(2):
                mapped_points.append({
                    'X': x,
                    'Y': y,
                    'assigned_ideal': ideal_name,
                    'deviation': deviation
                })
                break  # Assign to first matching ideal
    return pd.DataFrame(mapped_points)

def visualize(training_data, selected_ideals, ideal_data, test_mapped):
    plt.figure(figsize=(12, 8))

    # Plot training datasets
    for t_name, t_df in training_data.items():
        plt.scatter(t_df['X'], t_df['Y'], label=f"Training {t_name}", marker='o')

    # Plot selected ideal functions
    for train_name, ideal_name in selected_ideals.items():
        i_df = ideal_data[ideal_name]
        plt.plot(i_df['X'], i_df['Y'], label=f"Ideal for {train_name}", linewidth=2)

    # Plot mapped test points
    if not test_mapped.empty:
        for ideal_name in test_mapped['assigned_ideal'].unique():
            df = test_mapped[test_mapped['assigned_ideal'] == ideal_name]
            plt.scatter(df['X'], df['Y'], label=f"Test mapped to {ideal_name}", marker='x')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Training, Ideal Functions, and Mapped Test Points')
    plt.legend()
    plt.show()

def main():
    # Update these paths according to your folder structure
    training_folder = r"C:\C:\Users\Marzieh\PycharmProjects\PythonProject\training"
    test_folder = r"C:\Users\Marzieh\PycharmProjects\PythonProject\test"
    ideal_folder = r"C:\Users\Marzieh\PycharmProjects\PythonProject\ideal"

    training_data = load_csv_files(training_folder)
    test_data = load_csv_files(test_folder)
    ideal_data = load_csv_files(ideal_folder)

    selected_ideals, max_devs = select_best_ideals(training_data, ideal_data)

    # Assuming one test dataset
    test_df = list(test_data.values())[0]
    test_mapped = map_test_points(test_df, selected_ideals, ideal_data, max_devs)

    # Save mapped test points to CSV
    test_mapped.to_csv('mapped_test_points.csv', index=False)
    print("Mapped test points saved to 'mapped_test_points.csv'.")

    visualize(training_data, selected_ideals, ideal_data, test_mapped)

if __name__ == "__main__":
    main()
