from ultralytics import YOLO
from PIL import Image

model = YOLO("best.pt")

# Load and test an image that exists
img = Image.open("test_images/images.jpeg")  # <-- use a real path

results = model(img)

# Print results as a DataFrame
print(results[0].to_df())  # Show boxes, confidences, class names

# OR print results as JSON
# print(results[0].to_json())

# Optional: show annotated image
results[0].show()

# Optional: save the image with results
results[0].save(filename="backend/output.jpg")