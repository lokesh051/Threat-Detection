# Threat-Detection
This project leverage real-time object detection to detect threat. In this project I have added 2 objects Knife and Scissors in a list potentially dangerous objects. Current System will play a sound whenever it will detect either of these two objects to alert a user of a potential threat.

## Dataset & Algorithm
#### Dataset: ssd_mobilenet v2 coco
#### Detection Algorithm: (yolo-v3) [https://github.com/mystic123/tensorflow-yolo-v3.git]

## Steps to run this project

1. Install [Open Vino](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_macos.html) Toolkit for your OS 
2. Make sure to initialize your openvino environment(refer the documentation) 
3. Open Terminal and run
```
cd ~/intel/openvino/deployment_tools/model_optimizer/
```
4. Generate the IR of the YOLOv3 TensorFlow model, run:
```
python3 mo_tf.py
--input_model /path/to/yolo_v3.pb
--tensorflow_use_custom_operations_config $MO_ROOT/extensions/front/tf/yolo_v3.json
--batch 1
```
5. Now feed the IR to the IE
```
python app.py -m /path/to/frozen_darknet_yolov3_model -ct 0.4
```
*Note: Do not add file extension after 'frozen_darknet_yolov3_model'*
