import numpy as np
import cv2
import face_recognition
import sys
from torchvision import transforms
from PIL import Image
from model import predict_emotion

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor()
])

video = cv2.VideoCapture(-1)  # Open a video with the first available camera device

# Close if no camera device detected
if not video.isOpened():
    sys.exit("!!!---Camera not detected---!!!")

# Loop over various frames to create an image
while True:
    _, frame = video.read()  # Read video from camera
    rgb_frame = frame[:, :, ::-1]  # Taking tha channel and reversing it from BGR to RGB

    # Close program, If no image is collected
    if _ is False:
        sys.exit("!!!---Image not detected---!!!")

    face_locations = face_recognition.face_locations(rgb_frame)  # Detect locations of frame
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)  #

    for top, right, bottom, left in face_locations:
        show_feed = cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0),
                      2)  # Adds the rectangle around detected location of the faces

        face_crop = rgb_frame[top:bottom, left:right, :]  # Lets Get the Face Part, so we can make predictions on them
        cv2.imshow("Face Crop", face_crop)
        pil_image = Image.fromarray(face_crop)  # Convert Numpy Array to PIL Image
        transformed_pil_image = transform(pil_image)  # Apply transformations on PIL Image
        transformed_pil_image = transformed_pil_image.reshape([1, 3, 224, 224])
        # print(transformed_pil_image)
        emotion_predictions = predict_emotion(data=transformed_pil_image)
        emotion_predictions = f"{emotion_predictions}"
        emotion_predictions = emotion_predictiqons.replace(",", "\n")
        print(emotion_predictions)
        cv2.putText(show_feed, emotion_predictions, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36, 255, 12), 2)

    cv2.imshow("Face Detection", frame)  # Show the Image

    # Use q to close
    if cv2.waitKey(1) == ord("q"):
        break

video.release()  # release camera device
cv2.destroyAllWindows()  # Close all open windows
