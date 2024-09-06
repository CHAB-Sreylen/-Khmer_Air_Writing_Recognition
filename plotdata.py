import matplotlib.pyplot as plt

def parse_line(line):
    # Split the line by commas and convert to float
    coordinates = [float(coord) for coord in line.strip().split(',') if coord]
    return coordinates
     
    
# Replace your file here
data_file = r"D:\I4-internship\Internship\data\train2.csv"






data = []


with open(data_file, 'r', encoding='utf-8') as file:
    for line in file:
        if line.strip() == "":
            continue  # Skip empty lines
        coordinates = parse_line(line)
        if coordinates:  # Only add non-empty lists
            data.append(coordinates)

# Plotting all data in one graph
plt.figure(figsize=(8, 6))


for sequence in data:
    x = sequence[0::2]  # Take every even index as x-coordinate
    y = sequence[1::2]  # Take every odd index as y-coordinate
    plt.scatter(x, y)
    plt.plot(x, y)  # Connect points with a line to show sequence

plt.gca().invert_yaxis()  # Invert y-axis if needed for correct orientation
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Plot of Coordinates from CSV')
plt.grid(True)
plt.show()




# path = '/content/drive/MyDrive/Intern_Year4/Object_Detection/simple_obj_detect/test/annotations/a (1).xml'
# result = get_bounding_box(path