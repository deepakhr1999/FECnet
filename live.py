from tensorflow.keras.models import load_model
import cv2
import numpy as np
import skimage

model = load_model('models/emotions.h5')
aligner = cv2.CascadeClassifier('models/haarcascade_frontalface_alt2.xml')
targets = ['Anger',  'Dislike',  'Fear',  'Happy',  'Neutral',  'Sad',  'Surprised']

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for (x, y, w, h) in aligner.detectMultiScale(gray):
        m = int(w*0.05)
        x,y,w,h = x-m, y-m, w+m, h+m
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image = cv2.cvtColor(gray[y:y+h,x:x+w], cv2.COLOR_GRAY2RGB)
        image = skimage.transform.resize(image, (224,224)).astype('float32')/255
        image = image.reshape((1,*image.shape))
        i = model.predict(image).argmax()
        cv2.putText(frame, targets[i], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,0,0), 3, cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()