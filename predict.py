import torch
import torch.nn as nn
import numpy as np

# Convert into input format 
def format_input(points):
    return ', '.join(f'{x:.6f},{y:.6f}' for x, y in points if isinstance((x, y), tuple)).strip()


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist if item]


class ANNModel(nn.Module):
    def __init__(self, layers_sz, in_sz, out_sz):
        super().__init__()
        layers = [nn.Linear(in_sz, layers_sz[0])]
        layers += [nn.Linear(layers_sz[i], layers_sz[i + 1]) for i in range(len(layers_sz) - 1)]
        self.linears = nn.ModuleList(layers)
        self.out = nn.Linear(layers_sz[-1], out_sz)
        self.act_func = nn.LeakyReLU()

    def forward(self, x):
        for layer in self.linears:
            x = self.act_func(layer(x))
        return self.out(x)


def predict_char(formatted_strokes, model_path='model/Model_100_Epoch_cleaned_data_bz_300.pt'):
    stroke_list = np.array([float(x) for x in formatted_strokes.split(',')]).reshape(1, -1)
    expected_input_size = 2362

    # Adjust the input size
    stroke_list = np.pad(stroke_list, ((0, 0), (0, max(0, expected_input_size - stroke_list.shape[1]))), 'constant')
    stroke_list = stroke_list[:, :expected_input_size]
    x_test = torch.tensor(stroke_list, dtype=torch.float32)

    # Load model
    model = ANNModel(layers_sz=[1024, 512], in_sz=expected_input_size, out_sz=33)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        prediction = torch.argmax(model(x_test)).item()

    khmer_chars = {
        0: 'ក', 1: 'ខ', 2: 'គ', 3: 'ឃ', 4: 'ង', 5: 'ច', 6: 'ឆ', 7: 'ជ', 8: 'ឈ', 9: 'ញ',
        10: 'ដ', 11: 'ឋ', 12: 'ឌ', 13: 'ឍ', 14: 'ណ', 15: 'ត', 16: 'ថ', 17: 'ទ', 18: 'ធ', 19: 'ន',
        20: 'ប', 21: 'ផ', 22: 'ព', 23: 'ភ', 24: 'ម', 25: 'យ', 26: 'រ', 27: 'ល', 28: 'វ', 29: 'ស',
        30: 'ហ', 31: 'ឡ', 32: 'អ'
    }
    return khmer_chars.get(prediction)


# Example usage:
# formatted_strokes = '0.311851,0.465184, 0.312942,0.447050, ...'  # Example input
# predicted_char = predict_char(formatted_strokes)
# print(predicted_char)
