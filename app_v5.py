import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
import util 
from util import write_csv
import os
from sort.sort import *   
import av



folder_path = "./Detect_Images/"
LICENSE_MODEL_DETECTION_DIR = './models/best.pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"

reader = easyocr.Reader(['en'], gpu=True)


vehicles = [2, 3, 5, 7]
mot_tracker = Sort()



coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

threshold = 0.15

import string
import base64

import cv2
import ast
import numpy as np

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  #-- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  #-- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  #-- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img

def annotate_video_with_license_plates(frame, results,license_plate):
    if 'frame_nmr' not in results:
        return  # Exit if 'frame_nmr' key is not found in results

    for row_indx in range(len(results)):
        # Draw car
        car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(results.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                    line_length_x=200, line_length_y=200)

        # Draw license plate
        x1, y1, x2, y2 = ast.literal_eval(results.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

        # Crop license plate
        license_crop = license_plate[results.iloc[row_indx]['car_id']]['license_crop']
        H, W, _ = license_crop.shape

        try:
            frame[int(car_y1) - H - 100:int(car_y1) - 100,
                  int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

            frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                  int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

            (text_width, text_height), _ = cv2.getTextSize(
                license_plate[results.iloc[row_indx]['car_id']]['license_plate_number'],
                cv2.FONT_HERSHEY_SIMPLEX,
                4.3,
                17)

            cv2.putText(frame,
                        license_plate[results.iloc[row_indx]['car_id']]['license_plate_number'],
                        (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4.3,
                        (0, 0, 0),
                        17)

        except:
            pass

    # Display frame on GUI
    update_label(frame)

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def license_complies_format(text):
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """
    scores = 0
    detections = reader.readtext(license_plate_crop)

    if not detections:
        return None, None

    rectangle_size = license_plate_crop.shape[0] * license_plate_crop.shape[1]
    plate = []

    for result in detections:
        bbox, text, score = result
        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            text = format_license(text)
            scores += score
            plate.append(text)

    if plate:
        return " ".join(plate), scores / len(plate)
    else:
        return None, None



import os
import csv
import cv2
import uuid
from datetime import datetime

def save_results(license_plate_text, license_plate_crop, csv_filename, folder_path,real_time):
    # Generate a unique image name
    img_name = "{}.jpg".format(uuid.uuid1())
    
    # Save the license plate crop as an image
    cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)
    
    # Get the current time in the desired format
    real_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Append the results to the CSV file
    with open(csv_filename, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name, license_plate_text, real_time])


def get_car(license_plate, vehicle_track_ids):

    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1


previous_results = {}  # Initialize previous results
best_scores = {}  # Initialize best scores for each vehicle
best_texts = {}  # Initialize best texts for each vehicle
best_license_plate_crops = {}  # Initialize best license plate crops for each vehicle


def detect_license_plate(frame_nmr, frame, reader, coco_model, license_plate_detector, mot_tracker, vehicles, results):
    global previous_results, best_scores, best_texts, best_license_plate_crops

    # Ensure that results dictionary has an entry for the current frame number
    
    if frame_nmr not in results:
        results[frame_nmr] = {}

    # Detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id  = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])
            # class_name = ""  # Assign a default value outside the if statement
            # if int(class_id) == 2:
            #     class_name = "Car"
            # elif int(class_id) == 3:
            #     class_name = "Motor"
            # elif int(class_id) == 5:
            #     class_name = "Bus"
            # elif int(class_id) == 7:
            #     class_name = "Truck"
            # else:
            #     class_name = "Unknown"

    # Track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

    # Detect license plates
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        

        # Assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
        if car_id != -1:
            # Crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

            # Process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # Read license plate number
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            
            cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (255, 0, 255), thickness=2)

            # text_size, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # text_width, text_height = text_size
            # # Draw the text on the image
            # text_org = (int(xcar1) + 5, int(ycar1) + 20)  # Adjust the text position
            # cv2.putText(frame, class_name, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # white box 
            cv2.rectangle(frame, (int(x1) - 25, int(y1) - 25), (int(x2) + 25, int(y1)), (255, 255, 255), cv2.FILLED)

            if license_plate_text is not None:
                results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                              'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                'text': license_plate_text,
                                                                'bbox_score': score,
                                                                'text_score': license_plate_text_score}}
            
                resized_license_plate_crop = cv2.resize(license_plate_crop, (220, 60), interpolation=cv2.INTER_LINEAR)

                # Get the image region corresponding to the license plate
                rgb_region = cv2.cvtColor(resized_license_plate_crop, cv2.COLOR_BGR2RGB)
                img_tk_region = ImageTk.PhotoImage(Image.fromarray(rgb_region))

                # Display the result with the best text for the current vehicle
                display_result(license_plate_text, img_tk_region)

                # Save the resized license plate crop and result
                save_results(license_plate_text, resized_license_plate_crop, "detection_results.csv", "Detection_Images", real_time="YYYY-MM-DD HH:MM:SS")

                
                cv2.putText(frame,
                            str(license_plate_text),
                            (int((int(x1) + int(x2)) / 2) - 30, int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 0),
                            2)

                # Display frame on GUI
                update_label(frame)  # Assuming update_label updates the GUI window with the frame

    # Annotate video with license plates
    # annotate_video_with_license_plates(frame, results, license_plate)


    # Write results to CSV file
    write_csv(results, './test.csv')


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
# Function to update the label with a new image
def update_label(image_np):
    # Resize the frame to a smaller size (e.g., 640x480)
    # resized_frame = cv2.resize(image_np, (640, 480))

    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # Convert the RGB frame to a Tkinter-compatible PhotoImage
    img_tk = ImageTk.PhotoImage(Image.fromarray(rgb_frame))

    # Update the label with the new image
    label.config(image=img_tk)
    label.image = img_tk


    
def resize_image(image, max_size):
    # Get the dimensions of the image
    height, width, _ = image.shape
    
    # Determine the maximum dimension
    max_dimension = max(height, width)
    
    # Calculate the scaling factor
    scale = max_size / max_dimension
    
    # Resize the image
    resized_image = cv2.resize(image, (int(width * scale), int(height * scale)))
    
    return resized_image


def upload_photo():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    frame_nmr = -1 
    if file_path:
        image = cv2.imread(file_path)

        frame_nmr += 1 
        
        # Resize the image for display with a maximum size of 640 pixels
        resized_image = resize_image(image, 640)

        # Detect license plate and get the processed image, license plate texts, and cropped license plate regions
        detect_license_plate(frame_nmr, resized_image, reader, coco_model, license_plate_detector, mot_tracker, vehicles, results)
        
        # Update the label with the processed image
        update_label(resized_image)

        # Use license plate texts and cropped regions as needed (e.g., save to file, display in another widget, etc.)
        # print("License Plate Texts:", licenses_texts)
        # print("License Plate Crops Total:", license_plate_crops_total)


# def upload_image():
#     # Implement logic for image mode
#     # For example, open a file dialog to select an image file
#     file_path = tk.filedialog.askopenfilename()
#     if file_path:
#         # Load the image and perform any processing or display it
#         image = Image.open(file_path)

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------- 

cap = None

def stop_current_process():
    global cap, stop_camera
    if cap is not None and cap.isOpened():
        cap.release()
    stop_camera = True

results = {}  # Initialize results globally



def upload_video():
    global cap, stop_camera, frame_nmr  # Define frame_nmr as global
    frame_nmr = -1  # Initialize frame_nmr
    stop_camera = False  # Reset the stop_camera flag
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    print("Selected file:", file_path)  # Debugging: Print selected file path
    if file_path:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("Error: Unable to open video file")
            return

        # Define the desired display size
        display_width = 700
        display_height = 430

        while cap.isOpened() and not stop_camera:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_nmr += 1  # Increment frame number

            # Resize the frame to the desired display size
            resized_frame = cv2.resize(frame, (display_width, display_height))

            # Perform detection on the resized frame
            
            detect_license_plate(frame_nmr, resized_frame, reader, coco_model, license_plate_detector, mot_tracker, vehicles, results)
           
            # Update the label with the resized image
            update_label(resized_frame)
          
            # Update the window to display the new image
            window.update() 

        cap.release()
        cv2.destroyAllWindows()



#---------------------------------------------------------------------------------------------------------------------

 
import os
import cv2
import numpy as np
import tkinter as tk
from pathlib import Path
from tkinter import *
from tkinter import filedialog
from tkinter.font import Font
from tkinter import  ttk, END
from tkinter import  Canvas, Entry, Text, Button, PhotoImage
from tkinter import  ttk, Label, PhotoImage
from datetime import datetime
from PIL import Image, ImageTk


OUTPUT_PATH = os.path.dirname(os.path.realpath('__file__'))
ASSETS_PATH = OUTPUT_PATH / Path(r"S:\Indonesia\skripsi\ANPR_YOLOv8\GUI")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Create the main GUI window
window = tk.Tk()
window.geometry("1700x823")  # Adjust the width as needed
window.configure(bg="#FFFFFF")
window.title("Result Display")

# Create a frame for the application on the left side
app_frame = ttk.Frame(window, width=650, height=820)  # Adjust the width and height as needed
app_frame.grid(row=0, column=0, padx=200, pady=10, sticky="nsew")

style = ttk.Style()

# Set the background color for the Treeview
style.configure('Treeview', rowheight=70, background='#F0F0F0')  # Adjust the background color as needed

result_tree = ttk.Treeview(window, columns=("Image", "Text", "Time"))

result_tree.heading("#0", text="License Plate Image", anchor=tk.CENTER)
result_tree.heading("#1", text="License Plate Text", anchor=tk.CENTER)
result_tree.heading("#2", text="Time of Detection", anchor=tk.CENTER)
result_tree.heading("#3", text="", anchor=tk.CENTER)  # Hide the default column


result_tree.column("#0", width=300, anchor=tk.CENTER)
result_tree.column("#1", width=180, anchor=tk.CENTER)
result_tree.column("#2", width=150, anchor=tk.CENTER)
result_tree.column("#3", width=0, stretch=tk.NO)  # Hide the default column


# Place the Treeview in the window
result_tree.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

# Create a vertical scrollbar
scrollbar = Scrollbar(window, orient=VERTICAL, command=result_tree.yview)
scrollbar.grid(row=0, column=2, padx=10, pady=10, sticky="ns")

# Configure the Treeview to use the scrollbar
result_tree.configure(yscrollcommand=scrollbar.set)

# Counter for result IDs
result_id_counter = 1

# Global dictionary to store references to PhotoImage objects
photoimage_references = {}

class CustomPhotoImage:
    def __init__(self, image=None):
        if image:
            self._pil_image = Image.open(image)
            self._resize_image()
        else:
            self._pil_image = None
            self._photo_image = PhotoImage()

    def _resize_image(self):
        if self._pil_image:
            # Resize the image without maintaining aspect ratio
            new_size = (250, 50)  # Fixed size (width, height)
            resized_img_pil = self._pil_image.resize(new_size, Image.ANTIALIAS)
            self._photo_image = ImageTk.PhotoImage(resized_img_pil)

    def get_photo_image(self):
        return self._photo_image

    def resize(self, new_size):
        if self._pil_image:
            # Resize the image to the specified new size
            resized_img_pil = self._pil_image.resize(new_size, Image.ANTIALIAS)
            self._photo_image = ImageTk.PhotoImage(resized_img_pil)
            return self._photo_image
        else:
            return self._photo_image  # No resizing for empty PhotoImage
def display_result(license_plate_text, img_tk_region):
    global result_id_counter

    # Get the current time for the time of detection
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create a CustomPhotoImage instance
    original_img = CustomPhotoImage()

    # Set the existing PhotoImage to the CustomPhotoImage instance
    original_img._photo_image = img_tk_region

    # Resize the image within the CustomPhotoImage instance
    resized_img_tk = original_img.resize((250, 50))  # Fixed size (width, height)

    # Ensure that the resized image is retained as a reference
    photoimage_references[result_id_counter] = resized_img_tk

    # Create a new row with fixed height at the beginning of the Treeview
    result_tree.insert(parent="",
                       index=0,  # Insert at the beginning
                       iid=str(result_id_counter),
                       values=(license_plate_text, current_time),
                       image=resized_img_tk,
                       tags=str(result_id_counter))  # Set dynamic row height

    # Increment the result ID counter
    result_id_counter += 1

# Create a label to display the image
label = Label(window, bg="#F1F5FF")
label.place(x=300, y=150)

# Add a button to trigger extraction
button_extract = Button(
    text="Extract",
    borderwidth=0,
    highlightthickness=0,
    command='',
    relief="flat"
)
button_extract.place(x=594.0, y=640.0, width=126.0, height=61.0)


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 823,
    width = 1047,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    13.0,
    0.0,
    277.0,
    823.0,
    fill="#FFFFFF",
    outline="")

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_upload_image = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=upload_photo,
    relief="flat"
)
button_upload_image.place(
    x=38.0,
    y=117.0,
    width=180.0,
    height=61.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_upload_vid = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=upload_video,
    relief="flat"
)
button_upload_vid.place(
    x=38.0,
    y=214.0,
    width=180.0,
    height=61.0
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_from_camera = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command='open_camera',
    relief="flat"
)
button_from_camera.place(
    x=38.0,
    y=311.0,
    width=180.0,
    height=61.0
)

button_image_web_cam = PhotoImage(
    file=relative_to_assets("web1.png")
)

button_web_cam = Button(
    image=button_image_web_cam,
    borderwidth=0,
    highlightthickness=0,
    command='capIsOpen',  
    relief="flat"
)

button_web_cam.place(
    x=38.0,
    y=415.0
   
)

button_image_4 = PhotoImage(
    file=relative_to_assets("button_4.png"))
button_extract = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command='extract_number_plate',
    relief="flat"
)
#button_4.bind("<Button-1>", handle_q_press) # bind button press event to handle_q_press function
button_extract.place(
    x=594.0,
    y=640.0,
    width=126.0,
    height=61.0
)

button_image_5 = PhotoImage(
    file=relative_to_assets("button_5.png"))
button_quit = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command='quit_window',
    relief="flat"
)
button_quit.place(
    x=920.0,
    y=728.0,
    width=82.0,
    height=61.0
)

button_image_6 = PhotoImage(
    file=relative_to_assets("button_6.png"))
button_refresh = Button(
    image=button_image_6,
    borderwidth=0,
    highlightthickness=0,
    command='capIsOpen',
    relief="flat"
)
button_refresh.place(
    x=62.0,
    y=736.0,
    width=127.0,
    height=46.0
)



entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    657.0,
    760.5,
    image=entry_image_1
)
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# create a placeholder image with a solid color
img = Image.new('RGB', (670, 400), color='#F1F5FF')
img_tk = ImageTk.PhotoImage(img)

# create a label to display the image
label = Label(window, image=img_tk)
label.place(x=300, y=150)

#output1 = text
entry_1 = Entry(
    bd=0,
    bg="#F1F5FF",
    fg="#000716",
    highlightthickness=0,
    
)
# create a custom font with a larger size
font = Font(size=25)

# set the custom font as the font for the Entry widget
entry_1.config(font=font)
entry_1.place(
    x=528.0,
    y=729.0,
    width=258.0,
    height=61.0
)
#entry_1.insert(END, output1)

canvas.create_text(
    21.0,
    23.0,
    anchor="nw",
    text="Automatic",
    fill="#000000",
    font=("Arial BoldMT", 45 * -1)
)

canvas.create_text(
    297.0,
    23.0,
    anchor="nw",
    text="Number Plate Recognition ANPR",
    fill="#000000",
    font=("Arial BoldMT", 45 * -1)
)

canvas.create_text(
    291.0,
    741.0,
    anchor="nw",
    text="Number Plate :",
    fill="#000000",
    font=("Arial BoldMT", 30 * -1)
)

window.resizable(False, False)
window.mainloop()