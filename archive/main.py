import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
before = cv2.imread("data/before.png")
after = cv2.imread("data/after.png")

# Resize to same size
before = cv2.resize(before, (512, 512))
after = cv2.resize(after, (512, 512))

# Convert to HSV (better for vegetation detection)
before_hsv = cv2.cvtColor(before, cv2.COLOR_BGR2HSV)
after_hsv = cv2.cvtColor(after, cv2.COLOR_BGR2HSV)

# Define green color range (vegetation)
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# Create vegetation masks
mask_before = cv2.inRange(before_hsv, lower_green, upper_green)
mask_after = cv2.inRange(after_hsv, lower_green, upper_green)

# Calculate forest cover percentage
forest_before = np.sum(mask_before > 0)
forest_after = np.sum(mask_after > 0)
total_pixels = mask_before.size

percent_before = (forest_before / total_pixels) * 100
percent_after = (forest_after / total_pixels) * 100
change = percent_after - percent_before

# Decide result
if change > 0:
    result = "AFFORESTATION 🌱"
else:
    result = "DEFORESTATION 🌳➡️❌"

# Change map
change_map = cv2.absdiff(mask_after, mask_before)
heatmap = cv2.applyColorMap(change_map, cv2.COLORMAP_JET)

# Save outputs
cv2.imwrite("outputs/change_map.png", change_map)
cv2.imwrite("outputs/heatmap.png", heatmap)

# Display results
plt.figure(figsize=(12,6))

plt.subplot(2,3,1)
plt.title("Before (2014)")
plt.imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2,3,2)
plt.title("After (2025)")
plt.imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2,3,3)
plt.title("Change Heatmap")
plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2,3,4)
plt.title("Forest Mask Before")
plt.imshow(mask_before, cmap="gray")
plt.axis("off")

plt.subplot(2,3,5)
plt.title("Forest Mask After")
plt.imshow(mask_after, cmap="gray")
plt.axis("off")

plt.subplot(2,3,6)
plt.text(0.1, 0.6,
         f"Before: {percent_before:.2f}%\nAfter: {percent_after:.2f}%\nChange: {change:.2f}%\nResult: {result}",
         fontsize=12)
plt.axis("off")

plt.tight_layout()
plt.show()
