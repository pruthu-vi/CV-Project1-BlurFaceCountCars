import cv2

face_cascade = cv2.CascadeClassifier("Resources/cascade_by_opencv.xml")

img = cv2.imread("Resources/test_image.jpg")

detect = face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=6)
#print(detect)
#[[268 164  30  30]] Only one face is detected

for face in detect:
    x,y,w,h = face

    # only go through face and blur
    img[y:y+h,x:x+w] = cv2.GaussianBlur(img[y:y+h,x:x+w],(15,15),cv2.BORDER_DEFAULT)
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)

    cv2.imshow("Output face", img)

cv2.waitKey(1000)

# Save final image
filename = "Results/save_blured_face.jpg"
cv2.imwrite(filename, img)

# Save for next task
#filename = "Resources/save_blured_face.jpg"
#cv2.imwrite(filename, img)