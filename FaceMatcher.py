import cv2
import numpy as np
from FaceDetector import FaceDetector


class FaceMatcher:
    def __init__(self):
        self.face_detector = FaceDetector()


        self.feature_detector = cv2.ORB_create(
            nfeatures=500,           
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.match_threshold = 50


    def compute_feature_similarity(self, face1, face2):
        """Compute similarity between two face images using ORB features"""
        if face1 is None or face2 is None:
            return 0, None, None, None
        
        face1_resized = cv2.resize(face1, (200, 200))
        face2_resized = cv2.resize(face2, (200, 200))

        gray1 = cv2.cvtColor(face1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(face2_resized, cv2.COLOR_BGR2GRAY)

        gray1 = cv2.equalizeHist(gray1)
        gray2 = cv2.equalizeHist(gray2)


        kp1, des1 = self.feature_detector.detectAndCompute(gray1, None)
        kp2, des2 = self.feature_detector.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return 0, None, None, None
        

        matches = self.matcher.match(des1, des2)

        good_matches = [m for m in matches if m.distance < self.match_threshold]

        good_matches = sorted(good_matches, key=lambda x: x.distance)[:50]

        if len(good_matches) > 0:
           similarity = len(good_matches) / max(len(kp1), len(kp2))
        else:
            similarity = 0
            
        return similarity, good_matches, kp1, kp2
    
    def draw_matches(self, face1, face2, kp1, kp2, good_matches):
        """Draw matching keypoints between two faces"""
        if face1 is None or face2 is None:
            return None
        
       
        face1_resized = cv2.resize(face1, (200, 200))
        face2_resized = cv2.resize(face2, (200, 200))
        
        
        match_img = cv2.drawMatches(
            face1_resized, kp1,
            face2_resized, kp2,
            good_matches, None,
            matchColor=(0, 255, 0),  
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return match_img
    
    def extract_face_region(self, image, face_bbox):
        """Extract face region from image using bounding box"""
        x, y, w, h = face_bbox
        return image[y:y+h, x:x+w]
