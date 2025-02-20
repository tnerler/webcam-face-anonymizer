import cv2
import mediapipe as mp

# We initialize the Mediapipe face detection object outside the loop.
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


# Read webcam
webcam = cv2.VideoCapture(0)

while True :
    ret, frame = webcam.read()
    if not ret : 
        break

    H, W, _ = frame.shape

    # convert BGR to RGB

    webcam_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


     # Detect face
    output = face_detection.process(webcam_rgb)
        
    if output.detections is not None : 

        for detection in output.detections : 
            location_data = detection.location_data

            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height


            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x1 + w)
            y2 = min(H, y1 + h)


            # Blur face
            if x2 - x1 > 0 and y2 - y1 > 0:

                # Blurring the face step
                frame[y1:y2, x1:x2] = cv2.blur(frame[y1:y2, x1:x2], (50, 50))

    cv2.imshow('frame', frame)
    
    # You can press the 'q' button to shut down.
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()
face_detection.close()