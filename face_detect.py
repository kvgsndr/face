import cv2

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def max_arc(faces): 
    mx=(0,0,0,0)
    if len(faces)>0:
        for f in faces: # legnagyobb területű kép
            if f[2]*f[3] > mx[2]*mx[3]:
                mx=f
    return mx

def detect_max_face_box(kamerakep):  
    (x,y,h,w)=(0,0,1,1)
    gray_image = cv2.cvtColor(kamerakep, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(100, 100))
    if len(faces) > 0:
        (x, y, w, h)=max_arc(faces)
        cv2.rectangle(kamerakep, (x, y), (x + w, y + h), (0, 255, 0), 1)
        return  (x, y, w, h) #  van arc
    return (0,0, kamerakep.shape[0], kamerakep.shape[1])

def kep_kivág(kamerakep, face_box):
    (x,y,h,w)=face_box
    x=x-w//5
    if x<0:
        x=0
    y=y-h//5
    if y<0:
        y=0
    
    return kamerakep[y:y+h+2*(h//5), x:x+w+2*(w//5)]


def start():
    video_capture = cv2.VideoCapture(0)
    #kep=[]
    while True:
        #kep.clear()
        result, video_frame = video_capture.read()  # videó - kép beolvasása
        if not result:  #ha nincs kép
            continue  
        
        face = detect_max_face_box( video_frame )  # a felismert arcok legnagyobbikat
        #cv2.imwrite("proba.jpg",kep_kivág(video_frame, face))
        cv2.imshow("Arc",kep_kivág(video_frame, face))
        cv2.imshow( "Kamera", video_frame )  
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    video_capture.release()
    cv2.destroyAllWindows() 


if __name__ == "__main__":
    start()
