from PIL import Image, ImageDraw
import face_recognition
import time
import numpy as np
import cv2
import pickle
import device

# Get Face Model from file
with open("FaceData.data", 'rb') as fr:
    all_face_data = pickle.load(fr)

known_face_names = []
known_face_encodings = []
for key in all_face_data.keys():
    known_face_encodings.append(all_face_data[key])
    known_face_names.append(key)

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
retval = video_capture.isOpened()

process_this_frame = True
runtime = 0

while runtime < 1:
    # Grab a single frame of video or read file
    if retval:
        ret, frame = video_capture.read()
    else:
        frame = cv2.imread('.\\testimage\\couple.jpg')

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Only process every other frame of video to save time
    if process_this_frame:
        start = time.time()
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        end = time.time()
        sum = end - start
        print("辨識時間: {0}s".format(sum))

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size

        # Draw a box around the face
        cv2.rectangle(small_frame, (left, top),
                      (right, bottom), (0, 0, 255), 4)

        # Draw a label with a name below the face
        cv2.rectangle(small_frame, (left, bottom - 15),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(small_frame, name, (left + 6, bottom - 3),
                    font, 1.0, (255, 255, 255), 1)

    # Display the resulting imageq
    cv2.imshow('Video', small_frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
