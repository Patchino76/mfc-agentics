import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# Base64 string
file_path = 'outputs/output.txt'
with open(file_path, 'r') as file: 
    img_b64 = file.read()

# Remove the prefix if it exists
if img_b64.startswith("data:image/png;base64,"):
    img_b64 = img_b64.split(",")[1]

# Decode the base64 string
img_data = base64.b64decode(img_b64)
img = Image.open(BytesIO(img_data))

# Display the image
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.show()
