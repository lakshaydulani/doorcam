import face_recognition
import cv2
import numpy as np
import os
import time


stored_face_encodings = []
stored_face_names = []

def draw_border(img, point1, point2, point3, point4, line_length, color):

    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    x4, y4 = point4    

    cv2.circle(img, (x1, y1), 3, (255, 0, 255), -1)    #-- top_left
    cv2.circle(img, (x2, y2), 3, (255, 0, 255), -1)    #-- bottom-left
    cv2.circle(img, (x3, y3), 3, (255, 0, 255), -1)    #-- top-right
    cv2.circle(img, (x4, y4), 3, (255, 0, 255), -1)    #-- bottom-right

    cv2.line(img, (x1, y1), (x1 , y1 + line_length), color, 2)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length , y1), color, 2)

    cv2.line(img, (x2, y2), (x2 , y2 - line_length), color, 2)  #-- bottom-left
    cv2.line(img, (x2, y2), (x2 + line_length , y2), color, 2)

    cv2.line(img, (x3, y3), (x3 - line_length, y3), color, 2)  #-- top-right
    cv2.line(img, (x3, y3), (x3, y3 + line_length), color, 2)

    cv2.line(img, (x4, y4), (x4 , y4 - line_length), color, 2)  #-- bottom-right
    cv2.line(img, (x4, y4), (x4 - line_length , y4), color, 2)

    return img

def loadStoredFaces():
    global stored_face_encodings, stored_face_names
    
    for file in os.listdir("./storedFaces"):
        if file.endswith(".npy"):
            with open(os.path.join('./storedFaces/',file), 'rb') as f:
                stored_face_encodings.append(np.load(f)[0])
                stored_face_names.append(os.path.splitext(file)[0])
    
if __name__ == '__main__' :

    loadStoredFaces()

    # used to record the time when we processed last frame
    prev_frame_time = 0
 
    # used to record the time at which we processed current frame
    new_frame_time = 0

    cam = cv2.VideoCapture(1)

    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))
   
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('demo.mp4', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         5, size)

    while True:
        ret, frame = cam.read()
        
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = "FPS: " + str(fps)

        width  = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        cv2.putText(frame, fps, (width-90,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0, 255), 2)

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_location, face_encoding in zip(face_locations, face_encodings):
            stored_face_distances = face_recognition.face_distance(stored_face_encodings, face_encoding)
            best_stored_match_index = np.argmin(stored_face_distances)

            color = (0, 0, 255)
            name = "Unauthorized"
            top, right, bottom, left = face_location

            if stored_face_distances[best_stored_match_index] < 0.65:
                color = (0, 255, 0)
                name = stored_face_names[best_stored_match_index]
                

            draw_border(frame, (left, top), (left, bottom), (right, top), (right, bottom), 15, color)
            cv2.rectangle(frame, (left, bottom + 10), (right, bottom + 40), color, -1)
            ratio = (right - left) / 260.0
            cv2.putText(frame, name, (left + 6, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX , ratio, (255, 255, 255),2)

            cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        result.write(frame)

        

    cam.release()
    result.release()
    cv2.destroyAllWindows()

