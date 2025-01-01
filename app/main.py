import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from urllib.request import urlopen
import json
import torchvision.models as models

# Load ImageNet class index
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
with urlopen(url) as response:
    class_idx = json.load(response)


class Params:
    def __init__(self):
        self.batch_size = 64
        self.name = "resnet_50"
        self.workers = 4
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.lr_step_size = 30
        self.lr_gamma = 0.1

    def __repr__(self):
        return str(self.__dict__)
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__

params = Params()
# Load the model

model = models.resnet50(pretrained=False)
# model = ResNet50(Bottleneck, [3, 4, 6, 3])
model_path = "model_3.pth"
model.fc = torch.nn.Linear(2048, 1000)
checkpoint = torch.load(model_path, map_location="cpu")
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model file '{model_path}' not found.")

model_state_dict = checkpoint["model"]  # Update this key if needed
model.load_state_dict(model_state_dict, strict=True)
model.eval()
# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict(image: Image.Image) -> str:
    """Predict the top 5 classes for the given image."""
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        # print(outputs)  # Debug: Inspect raw model outputs

        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top5_prob, top5_catid = torch.topk(probabilities, 5)

    result = []
    for i in range(5):
        class_id = top5_catid[0][i].item()
        class_name = class_idx[str(class_id)][1]
        probability = top5_prob[0][i].item()
        result.append(f"{class_name}: {probability:.4f}")

    return "\n".join(result)

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="ResNet-50 Classifier",
    description="Upload an image to classify it using ResNet-50. The top 5 predictions will be displayed.",
)

if __name__ == "__main__":
    demo.launch()
