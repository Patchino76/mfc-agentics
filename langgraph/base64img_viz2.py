import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# Generate a sample plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])

# Save the plot to a buffer
buf = BytesIO()
fig.savefig(buf, format='png')

# Encode the image to base64
img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

# Save the base64 string to a file
file_path = 'outputs/output.txt'
with open(file_path, 'w') as file:
    file.write(img_b64)

# Read the base64 string from the file
with open(file_path, 'r') as file: 
    img_b64 = file.read()

# Add padding if necessary
missing_padding = len(img_b64) % 4
if missing_padding:
    img_b64 += '=' * (4 - missing_padding)

# Decode the base64 string
try:
    img_data = base64.b64decode(img_b64)
    img = Image.open(BytesIO(img_data))
    
    # Display the image
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()
except base64.binascii.Error as e:
    print(f"Base64 decoding error: {e}")
except Image.UnidentifiedImageError as e:
    print(f"Cannot identify image file: {e}")
