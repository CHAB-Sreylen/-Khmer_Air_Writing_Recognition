import matplotlib.pyplot as plt

def parse_line(line):
    coordinates = [float(coord) for coord in line.strip().split(',') if coord]
    return coordinates
     

data_file = "data/save_strokes.csv"

data = []



with open(data_file, 'r', encoding='utf-8') as file:
    for line in file:
        if line.strip() == "":
            continue  
        coordinates = parse_line(line)
        if coordinates: 
            data.append(coordinates)

# Plotting all data in one graph
plt.figure(figsize=(8, 6))


for sequence in data:
    x = sequence[0::2]  
    y = sequence[1::2]  
    plt.scatter(x, y)
    plt.plot(x, y)  

plt.gca().invert_yaxis() 
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Plot of Coordinates from CSV')
plt.grid(True)
plt.show()
