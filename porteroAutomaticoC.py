import face_recognition
import cv2

#AGREGADO POR MI
import glob
import os

# Load pictures and learn how to recognize it. MEJORADO POR MI
listadoImagenes=glob.glob("img/*.*")
known_face_encodings=[]
known_face_names=[]
for i in listadoImagenes:
    imagen=face_recognition.load_image_file(str(i))
    known_face_encodings=known_face_encodings+[face_recognition.face_encodings(imagen)[0]]
    known_face_names = known_face_names+[os.path.splitext(os.path.basename(str(i)))[0]]

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
ball=0
while True:
    # Grab a single frame of video

    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    font = cv2.FONT_HERSHEY_DUPLEX
    ball = 0

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconocida/o"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        ball = frame[top:bottom, left:right]

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, top - 20), (right, top), (0, 0, 255), cv2.FILLED)

        

        cv2.putText(frame, name, (left + 6, top - 6), font, 0.4, (255, 255, 255), 1)

    # Display the resulting image
    cv2.putText(frame, 'CANTIDAD DE PERSONAS AUTORIZADAS: '+str(len(listadoImagenes)), (20, 20), font, 0.5, (255, 255, 255), 1)

    
    
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("img/persona.jpg" , ball)	

        

    cv2.imshow('Video2', ball)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
