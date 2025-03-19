import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:4])  # Extract up to the first conv layer

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image = Image.open("cat.jpg")
input_tensor = preprocess(image).unsqueeze(0)

with torch.no_grad():
    activations = model(input_tensor)

activations = activations.squeeze().cpu().numpy()
num_filters = activations.shape[0]
plt.figure(figsize=(12, 12))
for i in range(num_filters):
    plt.subplot(8, 8, i+1)
    plt.imshow(activations[i], cmap="viridis")
    plt.axis("off")
plt.show()