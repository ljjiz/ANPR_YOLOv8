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
# from app import detect_license_plate1
import av


folder_path = "./Detection_Images/"
LICENSE_MODEL_DETECTION_DIR = './models/best.pt'
COCO_MODEL_DIR = "./models/yolov8n.pt"

reader = easyocr.Reader(['en'], gpu=True)


vehicles = [2, 3, 5, 7]
mot_tracker = Sort()



coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

threshold = 0.15


# class VideoProcessor:
#     def recv(self, frame) :
#         img = frame.to_ndarray(format="bgr24")
#         img_to_an = img.copy()
#         img_to_an = cv2.cvtColor(img_to_an, cv2.COLOR_RGB2BGR)
#         license_detections = license_plate_detector(img_to_an)[0]

#         if len(license_detections.boxes.cls.tolist()) != 0 :
#             for license_plate in license_detections.boxes.data.tolist() :
#                 x1, y1, x2, y2, score, class_id = license_plate

#                 cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

#                 license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
            
#                 license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) 

#                 license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

#                 cv2.rectangle(img, (int(x1) - 40, int(y1) - 40), (int(x2) + 40, int(y1)), (255, 255, 255), cv2.FILLED)
#                 cv2.putText(img,
#                             str(license_plate_text),
#                             (int((int(x1) + int(x2)) / 2) - 70, int(y1) - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX,
#                             1,
#                             (0, 0, 0),
#                             3)

#         return av.VideoFrame.from_ndarray(img, format="bgr24")
    

# def read_license_plate(license_plate_crop, img):
#     scores = 0
#     detections = reader.readtext(license_plate_crop)

#     width = img.shape[1]
#     height = img.shape[0]
    
#     if detections == [] :
#         return None, None

#     rectangle_size = license_plate_crop.shape[0]*license_plate_crop.shape[1]

#     plate = [] 

#     for result in detections:
#         length = np.sum(np.subtract(result[0][1], result[0][0]))
#         height = np.sum(np.subtract(result[0][2], result[0][1]))
        
#         if length*height / rectangle_size > 0.17:
#             bbox, text, score = result
#             text = result[1]
#             text = text.upper()
#             scores += score
#             plate.append(text)
    
#     if len(plate) != 0 : 
#         return " ".join(plate), scores/len(plate)
#     else :
#         return " ".join(plate), 0
    
# def read_license_plate(license_plate_crop, img):
#     scores = 0
#     detections = reader.readtext(license_plate_crop)

#     width = img.shape[1]
#     height = img.shape[0]
    
#     if detections == []:
#         return None, None

#     rectangle_size = license_plate_crop.shape[0] * license_plate_crop.shape[1]

#     plate = [] 

#     for result in detections:
#         length = np.sum(np.subtract(result[0][1], result[0][0]))
#         height = np.sum(np.subtract(result[0][2], result[0][1]))
        
#         if length * height / rectangle_size > 0.17:
#             bbox, text, score = result
#             text = result[1]
#             text = text.upper()
#             # text = format_license(text)  
#             scores += score
#             plate.append(text)
    
#     if len(plate) != 0: 
#         return " ".join(plate), scores / len(plate)
#     else:
#         return " ".join(plate), 0


# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)
import string

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

    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

def read_license_plate(license_plate_crop, frame):
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

def model_prediction(img):
    license_numbers = 0
    results = {}
    licenses_texts = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    object_detections = coco_model(img)[0]
    license_detections = license_plate_detector(img)[0]

    if len(object_detections.boxes.cls.tolist()) != 0 :
        for detection in object_detections.boxes.data.tolist() :
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection

            if int(class_id) in vehicles :
                cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)
    else :
            xcar1, ycar1, xcar2, ycar2 = 0, 0, 0, 0
            car_score = 0

    if len(license_detections.boxes.cls.tolist()) != 0 :
        license_plate_crops_total = []
        for license_plate in license_detections.boxes.data.tolist() :
            x1, y1, x2, y2, score, class_id = license_plate

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

            img_name = '{}.jpg'.format(uuid.uuid1())
         
            cv2.imwrite(os.path.join(folder_path, img_name), license_plate_crop)

            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY) 

            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

            licenses_texts.append(license_plate_text)

            if license_plate_text is not None and license_plate_text_score is not None  :
                license_plate_crops_total.append(license_plate_crop)
                results[license_numbers] = {}
                
                results[license_numbers][license_numbers] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2], 'car_score': car_score},
                                                        'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                            'text': license_plate_text,
                                                                            'bbox_score': score,
                                                                            'text_score': license_plate_text_score}} 
                license_numbers+=1
          
        write_csv(results, f"./csv_detections/detection_results.csv")

        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        return [img_wth_box, licenses_texts, license_plate_crops_total]
    
    else : 
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [img_wth_box]


import os
import csv
import cv2
import uuid
from datetime import datetime

def save_results(license_plate_text, resized_license_plate_crop, csv_filename, folder_path,real_time):
    # Generate a unique image name
    img_name = "{}.jpg".format(uuid.uuid1())
    
    # Save the license plate crop as an image
    cv2.imwrite(os.path.join(folder_path, img_name), resized_license_plate_crop)
    
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

def detect_license_plate(frame, reader, coco_model, license_plate_detector, folder_path, vehicles):
    global previous_results, best_scores, best_texts, best_license_plate_crops

    results = {}
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id  = detection
        if int(class_id) in vehicles:
            detections_.append([ x1, y1, x2, y2, score])
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
            # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)


    # Track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))


    best_car_id = None
    best_license_plate_crops_list = []
    # Detect license plates
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        # Draw rectangle around license plate crop

        # Assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:
            # Crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
            # Process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            # Apply adaptive thresholding
            _, license_plate_thresh = cv2.threshold(license_plate_crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # text_size, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # text_width, text_height = text_size
            # #Draw the text on the image
            # text_org = (int(xcar1) + 5, int(ycar1) + 20)  # Adjust the text position

            # cv2.putText(frame, class_name, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (255, 0, 255), thickness=2)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Draw the white rectangle
            # cv2.rectangle(frame, (int(x1) - 25, int(y1) - 25), (int(x2) + 25, int(y1)), (255, 255, 255), cv2.FILLED)
            # Read license plate number
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_thresh,frame)
            if license_plate_text is not None:
                # Calculate weighted score considering both license plate score and text score
                weighted_score = 0.6 * score + 0.4 * license_plate_text_score
                # Only consider the result if it has a better score than previous results
                if car_id not in previous_results or weighted_score > previous_results[car_id]['weighted_score']:
                    previous_results[car_id] = {
                        'license_plate_text': license_plate_text,
                        'license_plate_text_score': license_plate_text_score,
                        'weighted_score': weighted_score,
                        'car_bbox': [xcar1, ycar1, xcar2, ycar2]
                    }
                    best_scores[car_id] = weighted_score
                    best_texts[car_id] = license_plate_text
                    best_license_plate_crops[car_id] = license_plate_crop

                    # Store the result
                    results[car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {'bbox': [x1, y1, x2, y2],
                                          'text': license_plate_text,
                                          'bbox_score': score,
                                          'text_score': license_plate_text_score}
                    }
                    # # Define the font scale for smaller text
                    # font_scale = 0.4  # Adjust the font scale as needed

                    # # Calculate the text size
                    # text_size = cv2.getTextSize(str(license_plate_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
                    # text_width, text_height = text_size

                    # # Calculate the position of the text
                    # text_x = int((x1 + x2 - text_width) / 2)  # Center the text horizontally
                    # text_y = int(y1 - 10)  # Adjust the vertical position of the text

                    
        
        # Display the best result if available
        best_car_id = None
        if results:
            best_car_id = max(results, key=lambda x: results[x]['license_plate']['text_score'])
            if best_car_id:
                # Resize the license plate crop to a fixed size with improved resolution
                resized_license_plate_crop = cv2.resize(best_license_plate_crops[best_car_id], (220, 60), interpolation=cv2.INTER_LINEAR)
                
                # # Draw the text with the smaller font size
                # cv2.putText(frame,
                #             str(license_plate_text),
                #             (text_x, text_y),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             font_scale,
                #             (0, 0, 0),
                #             2)
                
                # Get the image region corresponding to the license plate
                rgb_region = cv2.cvtColor(resized_license_plate_crop, cv2.COLOR_BGR2RGB)
                img_tk_region = ImageTk.PhotoImage(Image.fromarray(rgb_region))

                # Display the result with the best text for the current vehicle
                display_result(best_texts[best_car_id], img_tk_region)


                # Convert best license plate crop to a list
                best_license_plate_crop = best_license_plate_crops[best_car_id]
                best_license_plate_crops_list = [best_license_plate_crop]

                # Save the resized license plate crop and result
                save_results([best_texts[best_car_id]], resized_license_plate_crop, "detection_results.csv", "Detection_Images", real_time="YYYY-MM-DD HH:MM:SS")

                write_csv({best_car_id: results[best_car_id]}, "./detection.csv")
        else:
            best_license_plate_crops_list = []


    img_wth_box = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return [img_wth_box, [best_texts.get(best_car_id, "")], best_license_plate_crops_list]





# def detect_license_plate(frame, reader, coco_model, detect_fn, vehicles):
#     global previous_results, best_scores, best_texts, best_license_plate_crops

#     results = {}

#     # Detect vehicles
#     detections = coco_model(frame)[0]
#     detections_ = []
#     for detection in detections.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = detection
#         if int(class_id) in vehicles:
#             detections_.append([x1, y1, x2, y2, score])
#             class_name = ""  # Assign a default value outside the if statement
#             if int(class_id) == 2:
#                 class_name = "Car"
#             elif int(class_id) == 3:
#                 class_name = "Motor"
#             elif int(class_id) == 5:
#                 class_name = "Bus"
#             elif int(class_id) == 7:
#                 class_name = "Truck"
#             else:
#                 class_name = "Unknown"

#     # Track vehicles
#     track_ids = mot_tracker.update(np.asarray(detections_))

#     # Detect license plates
#     input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
#     detections = detect_fn(input_tensor)

#     num_detections = int(detections.pop('num_detections'))
#     detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
#     detections['num_detections'] = num_detections
#     detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

#     # Visualize detection results
#     # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     viz_utils.visualize_boxes_and_labels_on_image_array(
#         frame,
#         detections['detection_boxes'],
#         detections['detection_classes'] + 1,
#         detections['detection_scores'],
#         category_index,
#         use_normalized_coordinates=True,
#         max_boxes_to_draw=5,
#         min_score_thresh=0.8,
#         agnostic_mode=False
#     )
#     # Convert the RGB frame to a Tkinter-compatible PhotoImage
#     # img_tk = ImageTk.PhotoImage(Image.fromarray(frame))

#     # # Update the label with the new image
#     # label.config(image=img_tk)
#     # label.image = img_tk 

#     for box_idx in range(len(detections['detection_boxes'])):
#         box = detections['detection_boxes'][box_idx]
#         class_id = detections['detection_classes'][box_idx]
#         score = detections['detection_scores'][box_idx]

#         # Extract box coordinates
#         ymin, xmin, ymax, xmax = box
#         ymin = int(ymin * frame.shape[0])
#         xmin = int(xmin * frame.shape[1])
#         ymax = int(ymax * frame.shape[0])
#         xmax = int(xmax * frame.shape[1])

#         # Assign license plate to car
#         xcar1, ycar1, xcar2, ycar2, car_id = get_car([ymin, xmin, ymax, xmax], track_ids)

#         if car_id != -1:
#             # Crop license plate
#             license_plate_crop = frame[ymin:ymax, xmin:xmax, :]
            
#             license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#             _, license_plate_thresh = cv2.threshold(license_plate_crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#             text_size, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#             text_width, text_height = text_size
#             text_org = (int(xcar1) + 5, int(ycar1) + 20)

#             cv2.putText(frame, class_name, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#             cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), thickness=2)
#             cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

#             # Update detections dictionary with license plate bounding box information
#             detections['detection_boxes'] = np.concatenate((detections['detection_boxes'], [[ymin/frame.shape[0], xmin/frame.shape[1], ymax/frame.shape[0], xmax/frame.shape[1]]]), axis=0)
#             detections['detection_classes'] = np.concatenate((detections['detection_classes'], [class_id]), axis=0)
#             detections['detection_scores'] = np.concatenate((detections['detection_scores'], [score]), axis=0)

#             # Read license plate
#             license_plate_text, license_plate_text_score = read_license_plate(license_plate_thresh, frame)

#             if license_plate_text is not None:
#                 weighted_score = 0.6 * score + 0.4 * license_plate_text_score
#                 if car_id not in previous_results or weighted_score > previous_results[car_id]['weighted_score']:
#                     previous_results[car_id] = {
#                         'license_plate_text': license_plate_text,
#                         'license_plate_text_score': license_plate_text_score,
#                         'weighted_score': weighted_score,
#                         'car_bbox': [xcar1, ycar1, xcar2, ycar2]
#                     }
#                     best_scores[car_id] = weighted_score
#                     best_texts[car_id] = license_plate_text
#                     best_license_plate_crops[car_id] = license_plate_crop

#                     results[car_id] = {
#                         'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                         'license_plate': {'bbox': [xmin, ymin, xmax, ymax],
#                                         'text': license_plate_text,
#                                         'bbox_score': score,
#                                         'text_score': license_plate_text_score}
#                 }

#     # Display the best result if available
#     best_car_id = None
#     if results:
#         best_car_id = max(results, key=lambda x: results[x]['license_plate']['text_score'])
#         if best_car_id:
#             resized_license_plate_crop = cv2.resize(best_license_plate_crops[best_car_id], (220, 60),
#                                                     interpolation=cv2.INTER_LINEAR)
#             rgb_region = cv2.cvtColor(resized_license_plate_crop, cv2.COLOR_BGR2RGB)
#             img_tk_region = ImageTk.PhotoImage(Image.fromarray(rgb_region))
#             display_result(best_texts[best_car_id], img_tk_region)
#             cv2.rectangle(frame, (results[best_car_id]['license_plate']['bbox'][0],
#                                   results[best_car_id]['license_plate']['bbox'][1]),
#                           (results[best_car_id]['license_plate']['bbox'][2],
#                            results[best_car_id]['license_plate']['bbox'][3]), (0, 255, 0), 2)

#             best_license_plate_crop = best_license_plate_crops[best_car_id]
#             best_license_plate_crops_list = [best_license_plate_crop]

#             save_results11([best_texts[best_car_id]], resized_license_plate_crop, "detection_results.csv",
#                         "Detection_Images", real_time="YYYY-MM-DD HH:MM:SS")
#             write_csv({best_car_id: results[best_car_id]}, "./detection.csv")
#     else:
#         best_license_plate_crops_list = []

#     img_wth_box = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     return [img_wth_box, [best_texts.get(best_car_id, "")], best_license_plate_crops_list]



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
    if file_path:
        image = cv2.imread(file_path)
        
        # Resize the image for display with a maximum size of 640 pixels
        resized_image = resize_image(image, 640)

        # Detect license plate and get the processed image, license plate texts, and cropped license plate regions
        processed_image, licenses_texts, license_plate_crops_total =  detect_license_plate(resized_image, reader, coco_model, license_plate_detector, folder_path, vehicles)
        
        # Check if any license plates are detected
        if licenses_texts:
            frame = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            # Update the label with the processed image
            update_label(frame)

            # Use license plate texts and cropped regions as needed (e.g., save to file, display in another widget, etc.)
            print("License Plate Texts:", licenses_texts)
            print("License Plate Crops Total:", license_plate_crops_total)
        else:
            print("No license plates detected in the image.")


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
    
def upload_video():
    global cap, stop_camera  # Access the global cap and stop_camera variables
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
        display_height = 394
        #394
        #430
        # display_width = 1300
        # display_height = 731
        #2560x1440
        while cap.isOpened() and not stop_camera:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize the frame to the desired display size
            resized_frame = cv2.resize(frame, (display_width, display_height))
            


            # Perform detection on the resized frame
            try:
                processed_image, licenses_texts, license_plate_crops_total = detect_license_plate(resized_frame, reader, coco_model, license_plate_detector, folder_path, vehicles)
            except ValueError as e:
                # Handle the case where not enough values are unpacked
                processed_image = resized_frame  # Assign resized_frame directly
                licenses_texts = None
                license_plate_crops_total = None
            frame = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            # Update the label with the resized image
            update_label(frame)
            print(licenses_texts, license_plate_crops_total)
            # Update the window to display the new image
            window.update() 

        cap.release()
        cv2.destroyAllWindows()



#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
def open_camera():
    # Implement logic for camera mode
    # For example, use OpenCV to capture frames from the camera
    pass
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
def open_live():
    # Implement logic for live detection mode
    # For example, use OpenCV to capture frames from the camera
    # and perform real-time detection
    pass
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
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
ASSETS_PATH = OUTPUT_PATH / Path(r"S:\Indonesia\semester 5\CV\skripsi\GUI\build\assets\frame0")


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
def display_result(best_texts, img_tk_region):
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
                       values=(best_texts, current_time),
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



# entry_image_1 = PhotoImage(
#     file=relative_to_assets("entry_1.png"))
# entry_bg_1 = canvas.create_image(
#     657.0,
#     760.5,
#     image=entry_image_1
# )
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# create a placeholder image with a solid color
img = Image.new('RGB', (670, 400), color='#F1F5FF')
img_tk = ImageTk.PhotoImage(img)

# create a label to display the image
label = Label(window, image=img_tk)
label.place(x=300, y=150)

# # #output1 = text
# entry_1 = Entry(
#     bd=0,
#     bg="#F1F5FF",
#     fg="#000716",
#     highlightthickness=0,
    
# )
# # create a custom font with a larger size
# font = Font(size=25)

# # set the custom font as the font for the Entry widget
# entry_1.config(font=font)
# entry_1.place(
#     x=528.0,
#     y=729.0,
#     width=258.0,
#     height=61.0
# )
# entry_1.insert(END, output1)

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