import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Device (CPU/GPU/MPS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Alphabetical list of 47 classes, as in dtd/images/
# Ensure this order matches the one used during training!
classes = [
    "banded", "blotchy", "braided", "bubbly", "bumpy", 
    "chequered", "cobwebbed", "cracked", "crosshatched", "crystalline",
    "dotted", "fibrous", "flecked", "freckled", "frilly", 
    "gauzy", "grid", "grooved", "honeycombed", "interlaced",
    "knitted", "lacelike", "lined", "marbled", "matted", 
    "meshed", "paisley", "perforated", "pitted", "pleated",
    "polka-dotted", "porous", "potholed", "scaly", "smeared",
    "spiralled", "sprinkled", "stained", "stratified", "striped",
    "studded", "swirly", "veined", "waffled", "woven",
    "wrinkled", "zigzagged"
]
num_classes = len(classes)  # 47

# Load pre-trained ResNet50
model = models.resnet50(pretrained=True)

# Modify the last layer for 47 classes (DTD)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# Load trained weights
# Assuming you saved them from the training script:
model.load_state_dict(torch.load("resnet50_dtd_split1.pth", map_location=device))
model.to(device)
model.eval()

# Transformations (as during training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# Load and prepare the image
image_path = "EXAMPLE.jpeg"  
image = Image.open(image_path).convert("RGB")

input_tensor = preprocess(image).unsqueeze(0)  
input_tensor = input_tensor.to(device)

# Pass through the network
with torch.no_grad():
    logits = model(input_tensor)            
    probs = F.softmax(logits, dim=1)      

# Get class index and name
pred_idx = torch.argmax(probs, dim=1).item()
predicted_class = classes[pred_idx]
confidence = probs[0, pred_idx].item()  # probability of the top class

print(f"Predicted class: {predicted_class} (confidence ~ {confidence:.2f})")
