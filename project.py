import PIL
import face_recognition
import numpy
from PIL.Image import Image
import speech_recognition as sr
import pyaudio
import pyttsx3
from PIL import Image, ImageDraw
#intalizing pyttsx3 lib

engine=pyttsx3.init()
#female voice change
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id) 
engine.say("program have started")
print("welcome, program have started")
engine.say("welcome to virtual face recognition project")
print("welcome to virtual face recognition project")
engine.say("This project is made by Jasleen kaur and Harshit Dhimann")
print("This project is made by Jasleen kaur and Harshit Dhimann")
engine.say("this project is made using python")
engine.runAndWait()

#Initialize recognizer class (for recognizing the speech)

engine.say("what name of person you want to find in picture:")
print("what name of person you want to find in picture:")
engine.say("olivia , nancy , geeta ,jiya")
print("olivia , nancy ,geeta ,   jiya")
engine.runAndWait()
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Talk")
    audio = r.listen(source)
    print("Time over, thanks")

		
# recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
  
audio_text=r.recognize_google(audio).lower()+".jpg"
        # using google speech recognition
print("Text: "+audio_text)

# #faces shown from image
engine.say("now the total faces in picture are ")
engine.runAndWait()
image=face_recognition.load_image_file("office.jpg")
face_locations=face_recognition.face_locations(image)
for face_location in face_locations:
       top,right,bottom,left=face_location
       face_image=image[top:bottom,left:right]
       pil_image=PIL.Image.fromarray(face_image)
    
   #other things you need to do snipped
       pil_image.show()
       pil_image.close()


#now actually face recognition starts

	# Load a sample picture and learn how to recognize it.
engine.say("now finding face of  in image.")
print("now finding face of  in image.")
engine.runAndWait()
known = face_recognition.load_image_file(audio_text)
encoding = face_recognition.face_encodings(known)[0] 
	# Load an image with unknown faces
unknown_image = face_recognition.load_image_file("office.jpg") 
	# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations) 	
    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = PIL.Image.fromarray(unknown_image) 
	# Create a Pillow ImageDraw Draw instance to draw with
draw = PIL.ImageDraw.Draw(pil_image) 
	# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings): 
	# See if the face is a match for the known face(s)
	    matches = face_recognition.compare_faces([encoding], face_encoding) 
	# Use the known face with the smallest distance to the new face
face_distances = face_recognition.face_distance([encoding], face_encoding)
best_match_index = numpy.argmin(face_distances)
if matches[best_match_index]:

	# Draw a box around the face using the Pillow module
            draw.rectangle(((left - 20, top - 20), (right + 20, bottom + 20)), outline=(0, 255, 0), width=20) 
    # Remove the drawing library from memory as per the Pillow docs
engine.say("Now green square is drawn around face. Thank you")
print("Now green square is drawn around faces. Thank you")
engine.runAndWait()
pil_image.show()	
del draw 
   



