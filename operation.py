


import os
import csv

data_folder = 'D:\\I4-internship\\Internship\\data'
os.makedirs(data_folder, exist_ok=True)

# Initialize empty list for saving strokes
strokes_for_saving = []

def save_strokes_csv(strokes, filename):
    combined_stroke = []
    for stroke in strokes:
        combined_stroke.extend(stroke)  # Combine all strokes into a single sequence

    # Validate and format combined stroke data
    formatted_data = []
    for point in combined_stroke:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            x, y = point
            formatted_data.append(f'{x:.6f},{y:.6f}')
        else:
            print(f"Warning: Skipping invalid point {point}")

    # Construct the file path
    file_path = os.path.join(data_folder, filename)

    # Write to the CSV file if there is any valid data
    with open(file_path, 'w', newline='') as file:
        if formatted_data:  # Only write if there is any valid stroke data
            file.write(','.join(formatted_data) + '\n')
        else:
            
            file.write('\n')


def load_strokes_csv(filename):
    strokes = []
    file_path = os.path.join(data_folder, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                points = []
                for i in range(0, len(row), 2):  # Read pairs of x, y values
                    try:
                        x = float(row[i])
                        y = float(row[i+1])
                        points.append((x, y))
                    except (ValueError, IndexError) as e:
                        print(f"Skipping invalid point in row: {row}. Error: {e}")
                if points:
                    strokes.append(points)
    return strokes

# Initialize strokes with previously saved data
strokes_for_saving = load_strokes_csv('train1.csv')
