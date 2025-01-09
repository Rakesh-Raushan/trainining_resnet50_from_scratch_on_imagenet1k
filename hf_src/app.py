import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import json
from huggingface_hub import hf_hub_download

# Load ImageNet class labels
with open("imagenet_classes.json") as f:
    labels = json.load(f)

#expects the params thing:
class Params:
    def __init__(self):
        self.batch_size = 128
        self.name = "resnet_50_onecycle"
        self.workers = 12
        self.max_lr = 0.11  # Maximum learning rate as per lr finder
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.epochs = 100
        self.pct_start = 0.3  # Percentage of training where LR increases
        self.div_factor = 25.0  # Initial LR = max_lr/div_factor
        self.final_div_factor = 1e4  # Final LR = max_lr/final_div_factor

def load_model():
    # model_path = "checkpoint.pth"
    model_path = hf_hub_download(
        repo_id="Perpetualquest/resnet50_explorer_v1",
        filename="trained_resnet_50_acc1_7371_epochs30.pth"
    )
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        state_dict = {k.replace('model.',''):v for k,v in state_dict.items()}
    else:
        state_dict = checkpoint  # In case the file only has the state_dict
    model = resnet50()
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

def predict(image):
    # Load model (we'll cache it)
    model = load_model()
    
    # Preprocess image
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = transform(image).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    # Get top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    # Create result dictionary
    results = []
    for i in range(5):
        results.append((labels[str(top5_catid[i].item())], float(top5_prob[i])))
    
    return {label: conf for label, conf in results}

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=5),
    title="ResNet50 ImageNet Classifier",
    description="Upload an image to classify it with a custom-trained ResNet50 model.",
    examples=["example1.jpg", "example2.jpg"]  # Add some example images
)

iface.launch() 