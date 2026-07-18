import cv2
import numpy as np
from FaceMatcher import FaceMatcher

def main():
    face_matcher = FaceMatcher()
    
    print("Loading reference image...")
    reference_img = cv2.imread('MohamadRezaShajaryan.jpg')
    if reference_img is None:
        print("Error: Could not load reference image")
        return
    
    reference_faces = face_matcher.face_detector.detect_faces(reference_img)
    if not reference_faces:
        print("No face detected in reference image")
        return
    print(f"Found {len(reference_faces)} face(s) in reference image")
    ref_face_bbox = reference_faces[0]['bbox']
    ref_face = face_matcher.extract_face_region(reference_img, ref_face_bbox)
    face_detected = face_matcher.face_detector.draw_faces(reference_img, reference_faces, 
                                             color=(255, 0, 0), thickness=2)
    cv2.imshow("Reference Image", face_detected)
    
    print("Loading video...")
    cap = cv2.VideoCapture('shajaryan.mp4')
    if not cap.isOpened():
        print("Error: Could not open video")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames, {fps:.2f} FPS")
    
    ret, first_frame = cap.read()
    
    if not ret:
        print("Error: Could not read first frame")
        return
    
    print("Detecting faces in video frame...")
    video_faces = face_matcher.face_detector.detect_faces(first_frame)
    print(f"Found {len(video_faces)} face(s) in video frame")
    
    display_frame = first_frame.copy()
    
    if video_faces:
        video_face_bbox = video_faces[0]['bbox']
        video_face = face_matcher.extract_face_region(first_frame, video_face_bbox)
        
        face_video_detected = face_matcher.face_detector.draw_faces(display_frame, video_faces, 
                                             color=(255, 0, 0), thickness=2)
        
        cv2.imshow("Video Frame", face_video_detected)
        
        similarity, good_matches, kp1, kp2 = face_matcher.compute_feature_similarity(
            ref_face, video_face
        )
        
        print(f"Face similarity score: {similarity:.2f}")
        
        MIN_MATCHES = 10
        MIN_SIMILARITY = 0.10
        
        if similarity > MIN_SIMILARITY and good_matches and len(good_matches) > MIN_MATCHES:
            print("Face match confirmed!")
            
            match_img = face_matcher.draw_matches(ref_face, video_face, kp1, kp2, good_matches)
            cv2.imshow('Face Match Details', match_img)
            
        else:
            print("Face match NOT confirmed. Playing video anyway...")
    
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    ref_height, ref_width = reference_img.shape[:2]
    
    print("Starting video playback...")
    frame_count = 0
    
    while True:
        ret, video_frame = cap.read()   
        if not ret:
            print("Video ended or error reading frame")
            break
        
        frame_count += 1
        
        video_resized = cv2.resize(video_frame, (ref_width, ref_height))
        cv2.imshow('Augmented Reality', video_resized)
        
       
        if frame_count % 30 == 0:  
            print(f"Playing frame {frame_count}/{total_frames}")
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q'):  
            print("Quit requested")
            break
    
    print(f"Played {frame_count} frames")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
