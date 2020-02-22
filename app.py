import argparse
import cv2
from openvino.inference_engine import IENetwork, IEPlugin
import sys, os, cv2, time
import numpy as np, math
import imutils
from imutils import paths
from pygame import mixer

mixer.init()
mixer.music.load("die.wav")




#INPUT_STREAM = "test_video.mp4"
CPU_EXTENSION = "~/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"

danger = ["knife", "scissors"]

m_input_size = 416

yolo_scale_13 = 13
yolo_scale_26 = 26
yolo_scale_52 = 52

classes = 80
coords = 4
num = 3
anchors = [10,14, 23,27, 37,58, 81,82, 135,169, 344,319]

labelsPath = os.path.sep.join(["tensorflow-yolo-v3", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

label_text_color = (255, 255, 255)
label_background_color = (125, 175, 75)
box_color = (255, 128, 0)
box_thickness = 1

def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)
	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    #i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ###       2) The user choosing the color of the bounding boxes
    ct_desc = "The confidence threshold to use with the bounding box"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    #optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-ct", help=ct_desc, default=0.2)
    args = parser.parse_args()

    return args





def EntryIndex(side, lcoords, lclasses, location, entry):
    n = int(location / (side * side))
    loc = location % (side * side)
    return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)


class DetectionObject():
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    class_id = 0
    confidence = 0.0

    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.class_id = class_id
        self.confidence = confidence


def IntersectionOverUnion(box_1, box_2):
    width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
    height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)
    area_of_overlap = 0.0
    if (width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0):
        area_of_overlap = 0.0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin)
    box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin)
    area_of_union = box_1_area + box_2_area - area_of_overlap
    retval = 0.0
    if area_of_union <= 0.0:
        retval = 0.0
    else:
        retval = (area_of_overlap / area_of_union)
    return retval


def ParseYOLOV3Output(blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold, objects):

    out_blob_h = blob.shape[2]
    out_blob_w = blob.shape[3]

    side = out_blob_h
    anchor_offset = 0

    if len(anchors) == 18:   ## YoloV3
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0

    elif len(anchors) == 12: ## tiny-YoloV3
        if side == yolo_scale_13:
            anchor_offset = 2 * 3
        elif side == yolo_scale_26:
            anchor_offset = 2 * 0

    else:                    ## ???
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0

    side_square = side * side
    output_blob = blob.flatten()

    for i in range(side_square):
        row = int(i / side)
        col = int(i % side)
        for n in range(num):
            obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords)
            box_index = EntryIndex(side, coords, classes, n * side * side + i, 0)
            scale = output_blob[obj_index]
            if (scale < threshold):
                continue
            x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w
            y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h
            height = math.exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1]
            width = math.exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n]
            for j in range(classes):
                class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j)
                prob = scale * output_blob[class_index]
                if prob < threshold:
                    continue
                obj = DetectionObject(x, y, height, width, j, prob, (original_im_h / resized_im_h), (original_im_w / resized_im_w))
                objects.append(obj)
    return objects


def infer_on_video(args):
    args.ct = float(args.ct)

    ### TODO: Initialize the Inference Engine
    # plugin = Network()
    # ### TODO: Load the network model into the IE
    # plugin.load_model(args.m, args.d, CPU_EXTENSION)
    # net_input_shape = plugin.get_input_shape()

    model_xml = args.m + '.xml'
    model_bin = args.m + ".bin"

    plugin = IEPlugin(device=args.d)
    if "CPU" in args.d:
        plugin.add_cpu_extension("lib/libcpu_extension.dylib")

    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    exec_net = plugin.load(network=net)

    print("IR Sucessfully loaded into the Inference Engine")

    camera_width = 640
    camera_height = 480
    fps = ""
    framepos = 0
    frame_count = 0
    vidfps = 0
    skip_frame = 0
    elapsedTime = 0
    new_w = int(camera_width * m_input_size/camera_width)
    new_h = int(camera_height * m_input_size/camera_height)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    time.sleep(1)

    while cap.isOpened():
        t1 = time.time()

        ## Uncomment only when playing video files
        #cap.set(cv2.CAP_PROP_POS_FRAMES, framepos)

        ret, image = cap.read()
        if not ret:
            break

        resized_image = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
        canvas = np.full((m_input_size, m_input_size, 3), 128)
        canvas[(m_input_size-new_h)//2:(m_input_size-new_h)//2 + new_h,(m_input_size-new_w)//2:(m_input_size-new_w)//2 + new_w,  :] = resized_image
        prepimg = canvas
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
        outputs = exec_net.infer(inputs={input_blob: prepimg})


        objects = []

        for output in outputs.values():
            objects = ParseYOLOV3Output(output, new_h, new_w, camera_height, camera_width, 0.4, objects)

        # Filtering overlapping boxes
        objlen = len(objects)
        for i in range(objlen):
            if (objects[i].confidence == 0.0):
                continue
            for j in range(i + 1, objlen):
                if (IntersectionOverUnion(objects[i], objects[j]) >= 0.4):
                    if objects[i].confidence < objects[j].confidence:
                        objects[i], objects[j] = objects[j], objects[i]
                    objects[j].confidence = 0.0

        # Drawing boxes
        for obj in objects:
            if obj.confidence < args.ct:
                continue
            label = obj.class_id
            confidence = obj.confidence
            #if confidence >= 0.2:
            label_text = LABELS[label] + " (" + "{:.1f}".format(confidence * 100) + "%)"
            if LABELS[label] in danger:
                mixer.music.play()
            cv2.rectangle(image, (obj.xmin, obj.ymin), (obj.xmax, obj.ymax), box_color, box_thickness)
            cv2.putText(image, label_text, (obj.xmin, obj.ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_text_color, 1)

        cv2.putText(image, fps, (camera_width - 170, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("Result", image)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break
        elapsedTime = time.time() - t1
        fps = "(Playback) {:.1f} FPS".format(1/elapsedTime)

        ## frame skip, video file only
        #skip_frame = int((vidfps - int(1/elapsedTime)) / int(1/elapsedTime))
        #framepos += skip_frame

    cv2.destroyAllWindows()
    del net
    del exec_net
    del plugin



def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
