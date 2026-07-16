import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        profile_cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
        self.profile_cascade = cv2.CascadeClassifier(profile_cascade_path)
        
        if self.face_cascade.empty() or self.profile_cascade.empty():
            print("Warning: Could not load cascade classifiers")
        else:
            print("OpenCV face detector initialized")
    
    def detect_faces(self, image):
        if image is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = []
        
        # Detect frontal faces
        detected_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in detected_faces:
            faces.append({
                'bbox': (x, y, w, h),
                'type': 'frontal'
            })
        
        # If no frontal faces found, try profile faces
        if len(faces) == 0:
            # Detect profile faces (both left and right)
            profile_faces = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in profile_faces:
                faces.append({
                    'bbox': (x, y, w, h),
                    'type': 'profile'
                })
            
        
        return faces
    
    def draw_faces(self, image, faces, color=(0, 255, 0), thickness=2):
        img_copy = image.copy()
        for face in faces:
            x, y, w, h = face['bbox']
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
            
            # Optionally add face type label
            if 'type' in face:
                cv2.putText(
                    img_copy,
                    face['type'],
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
        return img_copy
        