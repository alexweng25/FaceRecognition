from PIL import Image, ImageDraw
import face_recognition
import time
import cv2
import pickle

All_Face_Encodings = {}

# Load a picture and learn how to recognize it.
obama_image = face_recognition.load_image_file(".\\trainimage\\obama.jpg")
All_Face_Encodings["Obama"] = face_recognition.face_encodings(obama_image)[0]
face_locations = face_recognition.face_locations(obama_image)

for face_location in face_locations:
    top, right, bottom, left = tuple(int(x*0.25) for x in face_location)
    # 繪製人臉識別區域(寬度=line_border，顏色=(0, 255, 0))
    line_border = 5
    rgb_image = cv2.cvtColor(obama_image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.25, fy=0.25)
    cv2.rectangle(rgb_image, (left, top), (right, bottom),
                  (0, 255, 0), 4, line_border)

# 顯示結果
cv2.imshow("Obama", rgb_image)
cv2.waitKey(2)

# Load a picture and learn how to recognize it.
biden_image = face_recognition.load_image_file(".\\trainimage\\biden.jpg")
All_Face_Encodings["Biden"] = face_recognition.face_encodings(biden_image)[0]
face_locations = face_recognition.face_locations(biden_image)
for face_location in face_locations:
    top, right, bottom, left = tuple(int(x*0.25) for x in face_location)
    # 繪製人臉識別區域(寬度=line_border，顏色=(0, 255, 0))
    line_border = 5
    rgb_image = cv2.cvtColor(biden_image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.25, fy=0.25)
    cv2.rectangle(rgb_image, (left, top), (right, bottom),
                  (0, 255, 0), 4, line_border)

# 顯示結果
cv2.imshow("Biden", rgb_image)
cv2.waitKey(2)

# Load a picture and learn how to recognize it.
load_image = face_recognition.load_image_file(".\\trainimage\\Me.jpg")
All_Face_Encodings["Me"] = face_recognition.face_encodings(load_image)[0]
face_locations = face_recognition.face_locations(load_image)

for face_location in face_locations:
    top, right, bottom, left = tuple(int(x*0.25) for x in face_location)
    # 繪製人臉識別區域(寬度=line_border，顏色=(0, 255, 0))
    line_border = 5
    rgb_image = cv2.cvtColor(load_image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.25, fy=0.25)
    cv2.rectangle(rgb_image, (left, top), (right, bottom),
                  (0, 255, 0), 4, line_border)
# 顯示結果
cv2.imshow("Me", rgb_image)
cv2.waitKey(2)

# Load a picture and learn how to recognize it.
# load_image = face_recognition.load_image_file(".\trainimage\\wife.jpg")
# All_Face_Encodings["Fei"] = face_recognition.face_encodings(load_image)[0]
# face_locations = face_recognition.face_locations(load_image)

# for face_location in face_locations:
#     top, right, bottom, left = tuple(int(x*0.25) for x in face_location)
#     # 繪製人臉識別區域(寬度=line_border，顏色=(0, 255, 0))
#     line_border = 5
#     rgb_image = cv2.cvtColor(load_image, cv2.COLOR_BGR2RGB)
#     rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.25, fy=0.25)
#     cv2.rectangle(rgb_image, (left, top), (right, bottom),
#                   (0, 255, 0), 4, line_border)

with open("FaceData.data", 'wb') as fw:
    pickle.dump(All_Face_Encodings, fw)

cv2.waitKey(0)
