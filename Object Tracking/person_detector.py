import time, random
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from collections import deque

from absl import app, flags, logging
from absl.flags import FLAGS

from yolov3_tf2.models import (YoloV3, YoloV3Tiny)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


flags.DEFINE_string('classes', './data/labels/coco.names', 'classes file path')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 320, 'resize images to')
flags.DEFINE_string('video', './data/video/test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

centre_points = [deque(maxlen=50) for _ in range(9999)]

def main(_argv):

    #Use GPU if exists
    GPU_avb = tf.config.experimental.list_physical_devices('GPU')
    if len(GPU_avb) > 0:
        tf.config.experimental.set_memory_growth(GPU_avb[0], True)
    
    # Define the parameters and variables
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    counter = []
    
    #Initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    #Select Yolo configuration
    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    #Load weights
    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    #Load classes
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    #Capture video from file
    try:
        video = cv2.VideoCapture(int(FLAGS.video))
    except:
        video = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
    
    fps = 0.0
    count = 0 
    while True:
        curr_obj_counter = 0
    
        _, image = video.read()

        if image is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break
        
        #Reorder channels from BGR to RGB
        image_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image_in = tf.expand_dims(image_in, 0)
        image_in = transform_images(image_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(image_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(image, boxes[0])
        features = encoder(image, converted_boxes)    
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]        

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            class_name = track.get_class()
            if str(class_name) == 'person':
                if int(track.track_id) not in counter:
                    counter.append(int(track.track_id))
                curr_obj_counter += 1
                bbox = track.to_tlbr()
                cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0), 2)
                cv2.putText(image, class_name + " ID :" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)

                #Find center point of bounding box
                box_center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
                centre_points[track.track_id].append(box_center)
                cv2.circle(image,  (box_center), 1, (0,255,0) , 5)

                #Plot the motion path
                for j in range(1, len(centre_points[track.track_id])):
                    if centre_points[track.track_id][j - 1] is None or centre_points[track.track_id][j] is None:
                        continue
                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    cv2.line(image,(centre_points[track.track_id][j-1]), (centre_points[track.track_id][j]),(0,255,0), thickness)

        
        # print fps on screen 
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        count = len(set(counter))
        cv2.putText(image, "FPS: {:.2f}".format(fps), (20, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, "Total Persons Detected: "+str(count),(20, 450),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
        cv2.putText(image, "Persons Visible: "+str(curr_obj_counter),(20, 500),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
        cv2.imshow('output', image)
        if FLAGS.output:
            out.write(image)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(converted_boxes) != 0:
                for i in range(0,len(converted_boxes)):
                    list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
            list_file.write('\n')

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
    video.release()
    if FLAGS.output:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
