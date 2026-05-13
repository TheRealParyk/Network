import cv2
from matplotlib import pyplot as plt

# Load image
img = cv2.imread("people.png")

# Check if image loaded
if img is None:
    print("Error: Image not found.")
    exit()

# Convert colors
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load Haar cascade from OpenCV data folder
face_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

# Detect faces
faces = face_data.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

print(faces)

# Draw circles around faces
for (x, y, width, height) in faces:
    center = (x + width // 2, y + height // 2)
    radius = width // 2
    cv2.circle(img_rgb, center, radius, (0, 255, 0), 5)

# Show image
plt.imshow(img_rgb)
plt.axis("off")
plt.show()
