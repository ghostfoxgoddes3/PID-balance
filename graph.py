import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Load the video
video_path = "/mnt/data/PID ball balance.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    raise ValueError("Error opening video file")

# Parameters for ball detection (may need adjustments)
lower_hsv = np.array([0, 100, 100])  # Adjust based on ball color
upper_hsv = np.array([10, 255, 255])

positions = []  # To store (x, y) positions of the ball
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    frame_count += 1

    # Convert frame to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to detect the ball based on color
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assume the largest contour is the ball
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            positions.append((frame_count, cx, cy))

cap.release()

# Convert data to numpy array for plotting
positions = np.array(positions)
frame_indices, x_positions, y_positions = positions.T

# Apply a smoothing filter (Savitzky-Golay) for local linearization of y_positions
smoothed_y = savgol_filter(y_positions, window_length=11, polyorder=2)  # Adjust window as needed

# Plot the refined trajectory
plt.figure(figsize=(8, 5))
plt.plot(frame_indices, smoothed_y, label="Posição Y (suavizada)", color='r')
plt.xlabel("Quadro")
plt.ylabel("Posição Y (pixels)")
plt.title("Trajetória da Bola ao Longo do Tempo")
plt.legend()
plt.grid()
plt.show()