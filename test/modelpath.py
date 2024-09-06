from mediapipe.tasks.python.text import text_classifier
from mediapipe.tasks.python.core import BaseOptions

# Define the base options, typically including the model file path
base_options = BaseOptions(model_asset_path='path_to_your_model.tflite')

# Set up text classifier options with the defined base options
options = text_classifier.TextClassifierOptions(base_options=base_options)

# Initialize the text classifier
classifier = text_classifier.TextClassifier.create_from_options(options)

# Classify a piece of text
text_to_classify = "This is an example sentence."
result = classifier.classify(text_to_classify)

# Print the classification results
print(result)
