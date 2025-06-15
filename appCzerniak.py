import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import gradio as gr
from torchvision.models import resnet18, ResNet18_Weights

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  
model.load_state_dict(torch.load("modelResNetCzerniak.pth", map_location=torch.device("cpu")))
model.eval()

dark_theme = gr.themes.Base(
    primary_hue="blue", 
    neutral_hue="gray",      
    font="sans"             
).set(
    body_background_fill="#121212", 
    block_background_fill="#1E1E1E",
    body_text_color="#ffffff", 
    button_primary_background_fill="#2d72d9",  
    button_primary_text_color="#ffffff"        
)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        classes = ["Łagodna zmiana", "Nowotworowa zmiana"]
        return {classes[i]: float(probs[0, i]) for i in range(2)}


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Klasyfikator zmian skórnych",
    description="Model ResNet18 klasyfikujący zmiany skórne jako łagodne lub nowotworowe. Nie zastępuje porady medycznej.",
    theme=dark_theme,
    submit_btn="Zatwierdź",
    clear_btn="Wyczyść"
)

if __name__ == "__main__":
    interface.launch()