import streamlit as st 
import cv2
from PIL import Image,ImageEnhance
import numpy as np 

face_cascade = cv2.CascadeClassifier('/Users/bluedragondimension/Desktop/AboutCoding/K09Project/copy_cv2/data/haarcascade_frontalface_default.xml')

def detect_faces(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(img, 1.1, 4)
	for (x, y, w, h) in faces:
		img[y:y+h,x:x+w] = cv2.blur(img[y:y+h,x:x+w],(30,30),cv2.BORDER_DEFAULT)
	return img,faces

def main():
	"""Face Detection App"""
	st.title("Face Burring by /13")
	st.text("Credit: Streamlit and OpenCV")

	st.subheader("Face Burring")
	image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

	if image_file is not None:
		our_image = Image.open(image_file)
		st.text("Original Image")
		# st.write(type(our_image))
		st.image(our_image)

	enhance_type = st.sidebar.radio("Enhance Type",["Original","Blurring"])

	if enhance_type == 'Blurring':
		new_img = np.array(our_image.convert('RGB'))
		blur_rate = st.sidebar.slider("Brightness",0.5,3.5)
		img = cv2.cvtColor(new_img,1)
		blur_img = cv2.GaussianBlur(img,(11,11),blur_rate)
		st.image(blur_img)

	elif enhance_type == 'Original':
		st.image(our_image,width=300)

	# Face Detection
	if st.button("Process"):
		result_img,result_faces = detect_faces(our_image)
		st.image(result_img)			

if __name__ == '__main__':
	main()
