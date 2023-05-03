import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from yolo_utils import infer_image, show_image

# from tkinter import *
import tkinter as ttk
from tkinter import filedialog
from PIL import Image, ImageTk

FLAGS = []

def detect(FLAGS, unparsed):

	# Download the YOLOv3 models if needed
	if FLAGS.download_model:
		subprocess.call(['./yolov3-coco/get_model.sh'])

	# Get the labels
	labels = open(FLAGS.labels).read().strip().split('\n')

	# Intializing colors to represent each label uniquely
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	# Load the weights and configutation to form the pretrained YOLOv3 model
	net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

	# Get the output layer names of the model
	layer_names = net.getLayerNames()
	layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
			
	# If both image and video files are given then raise error
	if FLAGS.image_path is None and FLAGS.video_path is None:
		print ('Neither path to an image or path to video provided')
		print ('Starting Inference on Webcam')

	# Do inference with given image
	if FLAGS.image_path:
		# Read the image
		try:
			img = cv.imread(FLAGS.image_path)
			height, width = img.shape[:2]
		except:
			raise 'Image cannot be loaded!\n\
							Please check the path provided!'

		finally:
			img, _, _, _, _, vehicle_count, PCU = infer_image(net, layer_names, height, width, img, colors, labels, FLAGS)
			# print(type(img))
			photo = Image.fromarray(img)
			photo.thumbnail((700, 600))

			# print(type(photo))
			photo = ImageTk.PhotoImage(photo)
			image_label.config(image=photo)
			image_label.image = photo

			PCU_value_label.config(text=f'PCU Value: {PCU}')
			vehicle_count_label.config(text=f'Vehicle Count: {vehicle_count}')
			

			# show_image(img)
		
		# return 1, img

	elif FLAGS.video_path:
		# Read the video
		try:
			vid = cv.VideoCapture(FLAGS.video_path)
			height, width = None, None
			writer = None
		except:
			raise 'Video cannot be loaded!\n\
							Please check the path provided!'

		finally:
			while True:
				grabbed, frame = vid.read()

				# Checking if the complete video is read
				if not grabbed:
					break

				if width is None or height is None:
					height, width = frame.shape[:2]

				frame, _, _, _, _ = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS)

				if writer is None:
					# Initialize the video writer
					fourcc = cv.VideoWriter_fourcc(*"MJPG")
					writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30, 
									(frame.shape[1], frame.shape[0]), True)


				writer.write(frame)

			print ("[INFO] Cleaning up...")
			writer.release()
			vid.release()
			# return 2, 0


	else:
		# Infer real-time on webcam
		count = 0

		vid = cv.VideoCapture(0)
		while True:
			_, frame = vid.read()
			height, width = frame.shape[:2]

			if count == 0:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
									height, width, frame, colors, labels, FLAGS)
				count += 1
			else:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
									height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
				count = (count + 1) % 6

			cv.imshow('webcam', frame)

			if cv.waitKey(1) & 0xFF == ord('q'):
				break
		vid.release()
		cv.destroyAllWindows()

	# return 3, 0


if __name__ == '__main__':
	
	# class model_details():
	# 	MODEL_PATH = './cfg/yolo-coco'
	# 	WEIGHTS = 'yolov3.weights'
	# 	CONFIG = 'yolov3.cfg'
	# 	LABELS = 'coco.names'
	# 	IMAGE_PATH = 'mangalore-images/1.jpeg'
	# 	VIDEO_PATH = 'mangalore-images/2.mp4'
	# 	VIDEO_OUTPUT_PATH = './output.avi'
	# 	CONFIDENCE = 0.5
	# 	THRESHOLD = 0.3
	# 	DOWNLOAD_MODEL = False
	# 	SHOW_TIME = False

	# paste this into a the GUI code.
	parser = argparse.ArgumentParser()

	parser.add_argument('-m', '--model-path',
		type=str,
		default='./cfg/',
		help='The directory where the model weights and \
			  configuration files are.')

	parser.add_argument('-w', '--weights',
		type=str,
		default='./cfg/yolo-coco/yolov3.weights',
		help='Path to the file which contains the weights \
			 	for YOLOv3.')

	parser.add_argument('-cfg', '--config',
		type=str,
		default='./cfg/yolo-coco/yolov3.cfg',
		help='Path to the configuration file for the YOLOv3 model.')

	parser.add_argument('-i', '--image-path',
		type=str,
		default='./images/1.jpg',
		help='The path to the image file')

	parser.add_argument('-v', '--video-path',
		type=str,
		help='The path to the video file')


	parser.add_argument('-vo', '--video-output-path',
		type=str,
        default='./output.avi',
		help='The path of the output video file')

	parser.add_argument('-l', '--labels',
		type=str,
		default='./cfg/yolo-coco/coco.names',
		help='Path to the file having the \
					labels in a new-line seperated way.')

	parser.add_argument('-c', '--confidence',
		type=float,
		default=0.5,
		help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

	parser.add_argument('-th', '--threshold',
		type=float,
		default=0.3,
		help='The threshold to use when applying the \
				Non-Max Suppresion')

	parser.add_argument('--download-model',
		type=bool,
		default=False,
		help='Set to True, if the model weights and configurations \
				are not present on your local machine.')

	parser.add_argument('-t', '--show-time',
		type=bool,
		default=False,
		help='Show the time taken to infer each image.')

	FLAGS, unparsed = parser.parse_known_args()

	# root = Tk()
	# frm = ttk.Frame(root, padding=10)
	# frm.grid()
	# ttk.Label(frm, text="Hello World!").grid(column=0, row=0)
	# ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)


	def select_file():
		filetypes = (("JPEG files","*.jpg"),("PNG files","*.png"),("All files","*.*"))
		file_path = filedialog.askopenfilename(filetypes=filetypes)
		# print("Selected file:", file_path)
		FLAGS.image_path = file_path

		display_image(file_path)

	def display_image(file_path):
		image = Image.open(file_path)
		image.thumbnail((700, 600))
		# print(f'selction: {type(image)}')
		# print(type(image))
		photo = ImageTk.PhotoImage(image)
		image_label.config(image=photo)
		image_label.image = photo
		filename_label.config(text="Selected file: " + file_path)


	root = ttk.Tk()

		# Set size for the GUI
	root.geometry("700x700")
	root.configure(background='white')
	root.title("Image Selector")

	# Create a label widget for the filename
	filename_label = ttk.Label(root, text="")
	filename_label.pack()

	# Create a label widget for the image
	image_label = ttk.Label(root)
	image_label.pack(expand=True)

	PCU_value_label = ttk.Label(root, text='PCU Value: ')
	PCU_value_label.pack()

	vehicle_count_label = ttk.Label(root, text='Vehicle Count: ')
	vehicle_count_label.pack()

	PCU_value_label_hint = ttk.Label(root, text='PCU stands for Passenger Car Unit,\nwhich is a unit of measure used in transportation engineering to represent \nthe space occupied by a vehicle on a roadway or intersection. It is a way to normalize different\ntypes of vehicles by comparing them to a standard passenger car.')
	PCU_value_label_hint.pack()


	# PCU_value_label = ttk.Label(root, text='PCU Value:')
	# PCU_value_label.pack()
	# PCU_value_label_hint = ttk.Label(root, text='PCU stands for Passenger Car Unit, which is a unit of measure used in transportation engineering to represent the space occupied by a vehicle on a roadway or intersection. It is a way to normalize different types of vehicles by comparing them to a standard passenger car.')
	# PCU_value_label_hint.pack()

	# Create a button frame
	button_frame = ttk.Frame(root)
	button_frame.pack(side="bottom", pady=10)

	# Create "Select Image" button widget
	select_button = ttk.Button(button_frame, text="Select Image", command=select_file)
	select_button.pack(side="left", padx=10)

	# Create "Resize Image" button widget
	resize_button = ttk.Button(button_frame, text="Detect Vehicles", command=lambda: detect(FLAGS, unparsed))
	resize_button.pack(side="left", padx=10)

	photo = None
	root.mainloop()


	# status, res = detect(FLAGS, unparsed)
	# if status == 1:
	# 	show_image(res)
	# else:
	# 	pass