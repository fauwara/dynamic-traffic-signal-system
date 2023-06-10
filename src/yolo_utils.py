import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
import pprint

def show_image(img):
    cv.imshow("Image", img)
    cv.waitKey(0)

def map_id_to_value(id, labels):
    return labels[id]

def calculate_PCU(vehicle_count):
    PCU_for_vehicle = {
        'motorbike': 0.5,
        'car': 1,
        'bus': 3,
        'truck': 3,
    }
    
    PCU_value = 0 
    for class_ in vehicle_count:
        if class_ in PCU_for_vehicle:
            print(class_)
            PCU_value += PCU_for_vehicle[class_]*vehicle_count[class_]
    
    return PCU_value

def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    # If there are any detections
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            
            # Get the unique color for this class
            color = [int(c) for c in colors[classids[i]]]

            # Draw the bounding box rectangle and label on the image
            cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
            cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def generate_boxes_confidences_classids(outs, height, width, tconf):
    boxes = []
    confidences = []
    classids = []

    # dictionary to store count of label
    # labels_count = {}
    # for index, label in enumerate(labels):
    #     labels_count[label] = {'index': index, 'count': 0}

    # pprint.pprint(labels_count)

    for out in outs:
        for detection in out:
            #print (detection)
            #a = input('GO!')
            
            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]
            
            # Consider only the predictions that are above a certain confidence level
            if confidence > tconf:
                # TODO Check detection
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Append to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids

def infer_image(net, layer_names, height, width, img, colors, labels, FLAGS, 
            boxes=None, confidences=None, classids=None, idxs=None, infer=True):
    

    if infer:
        # Contructing a blob from the input image
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), 
                        swapRB=True, crop=False)

        # Perform a forward pass of the YOLO object detector
        net.setInput(blob)

        # Getting the outputs from the output layers
        start = time.time()
        outs = net.forward(layer_names)
        end = time.time()

        if FLAGS.show_time:
            print ("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

        
        # Generate the boxes, confidences, and classIDs
        boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, FLAGS.confidence)
        
        # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
        idxs = cv.dnn.NMSBoxes(boxes, confidences, FLAGS.confidence, FLAGS.threshold)


        # print(classids)
        vehicle_count = {}
        for i in idxs:
            if map_id_to_value(classids[i], labels) in vehicle_count:
                vehicle_count[map_id_to_value(classids[i], labels)] += 1
            else:
                vehicle_count[map_id_to_value(classids[i], labels)] = 1
        
        PCU = calculate_PCU(vehicle_count)
        # pprint.pprint(vehicle_count)
        # print(f'number of vehicles: {len(boxes)}, {confidences}\n{classids}')
        
        # print(idxs)

    if boxes is None or confidences is None or idxs is None or classids is None:
        raise '[ERROR] Required variables are set to None before drawing boxes on images.'
        
    # Draw labels and boxes on the image
    img = draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels)

    return img, boxes, confidences, classids, idxs, vehicle_count, PCU