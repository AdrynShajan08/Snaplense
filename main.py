import cv2
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the filter image (example: sunglasses)
# filter_img = cv2.imread('sunglasses.png', -1)
filter_img = cv2.imread('sunglasses_nobg.png', -1)

# Function to overlay the filter on the face
def overlay_filter(face_img, filter_img, x, y, w, h):
    # Check if the filter image has an alpha channel
    if filter_img.shape[2] == 4:
        # If it has an alpha channel, use it
        alpha_filter = filter_img[:, :, 3] / 255.0
        filter_rgb = filter_img[:, :, :3]
    else:
        # If it doesn't have an alpha channel, create one with all ones (fully opaque)
        alpha_filter = np.ones((h, w), dtype=np.float32)
        filter_rgb = filter_img

    
    # # Resize the filter image to match the face size
    # filter_resized = cv2.resize(filter_img, (w, h))
    
    # # # Create a mask for the filter
    # # alpha_filter = filter_resized[:, :, 3] / 255.0
    # # filter_rgb = filter_resized[:, :, :3]
    
    # # Calculate region of interest on the face image
    # roi = face_img[y:y+h, x:x+w]
    
    # # Overlay the filter on the face ROI
    # # for c in range(3):
    # #     roi[:, :, c] = (1 - alpha_filter) * roi[:, :, c] + alpha_filter * filter_rgb[:, :, c]
    
    # for c in range(3):
    #     # roi[:, :, c] = (1 - alpha_filter) * roi[:, :, c] + alpha_filter * filter_resized[:, :, c]
    #     # Ensure that both the ROI and the resized filter image have the same dimensions
    #     if filter_resized.shape[0] == roi.shape[0] and filter_resized.shape[1] == roi.shape[1]:
    #         roi[:, :, c] = (1 - alpha_filter) * roi[:, :, c] + alpha_filter * filter_resized[:, :, c]
    #     else:
    #         print("Filter image dimensions do not match face ROI dimensions.")
    
    
    # # Resize the filter image to match the size of the region of interest on the face image
    # filter_resized = cv2.resize(filter_rgb, (w, h))
    
    # # Calculate region of interest on the face image
    # roi = face_img[y:y+h, x:x+w]
    
    # # Overlay the filter on the face ROI
    # for c in range(3):
    #     roi[:, :, c] = (1 - alpha_filter) * roi[:, :, c] + alpha_filter * filter_resized[:, :, c]

    
    # # Calculate aspect ratio of the ROI
    # aspect_ratio = w / h
    
    # # Resize the filter image to match the aspect ratio of the ROI
    # filter_resized = cv2.resize(filter_rgb, (int(h * aspect_ratio), h))
    
    # # Calculate the width and height after resizing
    # new_w = filter_resized.shape[1]
    # new_h = filter_resized.shape[0]
    
    # # Calculate the offset to center the resized filter image on the ROI
    # offset_x = int((w - new_w) / 2)
    # offset_y = int((h - new_h) / 2)
    
    # # Calculate region of interest on the face image
    # roi = face_img[y:y+h, x:x+w]
    
    # # Overlay the filter on the face ROI
    # for c in range(3):
    #     roi[offset_y:offset_y+new_h, offset_x:offset_x+new_w, c] = (
    #         (1 - alpha_filter) * roi[offset_y:offset_y+new_h, offset_x:offset_x+new_w, c] +
    #         alpha_filter * filter_resized[:, :, c] )
    
    
    # Resize the filter image to match the size of the region of interest on the face image
    filter_resized = cv2.resize(filter_rgb, (w, h))
    print(filter_resized)
    print(filter_resized.shape)
    
    # Calculate region of interest on the face image
    roi = face_img[y:y+h, x:x+w]
    print(roi.shape)
    
    # Overlay the filter on the face ROI
    for c in range(3):
        # Ensure that both the ROI and the resized filter image have the same dimensions
        if filter_resized.shape[0] == roi.shape[0] and filter_resized.shape[1] == roi.shape[1]:
            roi[:, :, c] = (1 - alpha_filter) * roi[:, :, c] + alpha_filter * filter_resized[:, :, c]
        else:
            print("Filter image dimensions do not match face ROI dimensions.")
    

    return face_img

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Apply filter to each detected face
    for (x, y, w, h) in faces:
        frame = overlay_filter(frame, filter_img, x, y, w, h)

    # Display the resulting frame
    cv2.imshow('Snapchat Filter', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
