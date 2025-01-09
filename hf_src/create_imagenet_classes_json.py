import json
from torchvision.models import ResNet50_Weights

# Get the class mappings
weights = ResNet50_Weights.DEFAULT
class_mapping = weights.meta["categories"]

# Create a dictionary of index to class name
class_idx = {str(i): label for i, label in enumerate(class_mapping)}

# Save to JSON file
with open("imagenet_classes.json", "w") as f:
    json.dump(class_idx, f)