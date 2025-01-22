import cv2

face_cascade = cv2.CascadeClassifier('/home/abhishek-kumar/code/python/program/haarcascade_fullbody.xml')

def detect():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, img = cap.read()  # Unpack the frame
        if not ret:
            print("Failed to grab frame")
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Show the image with rectangles
        cv2.imshow("Face Detect", img)
        
        # Exit on pressing the 'Esc' key
        if cv2.waitKey(1) == 27:
            break

    # Release the capture and close any open windows
    cap.release()
    cv2.destroyAllWindows()

detect()
