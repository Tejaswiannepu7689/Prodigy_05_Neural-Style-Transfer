import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load images
content_image = Image.open("content.jpg").convert("RGB")
style_image = Image.open("style.jpg").convert("RGB")

# Resize images
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

content_tensor = transform(content_image)
style_tensor = transform(style_image)

# Placeholder for Neural Style Transfer output
output_tensor = content_tensor

# Display images
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(content_image)
plt.title("Content Image")

plt.subplot(1,3,2)
plt.imshow(style_image)
plt.title("Style Image")

plt.subplot(1,3,3)
plt.imshow(output_tensor.permute(1,2,0))
plt.title("Stylized Output")

plt.show()