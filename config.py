################
# Model Weight #
################
yolo_weight_path = 'weight/yolov7-w6-person.pt'
reid_weight_path = 'weight/osnet_ain_x1_0.pth'


#####################################
# Model Hyperparatmer Configuration #
#####################################

input_size = 640

yolo = {
    "conf_thres": 0.70,  # confidence threshold
    "iou_thres": 0.55,  # NMS IOU threshold
    "agnostic_nms": False,
    "max_det": 75, # maximum detections per image
}

matching = {
    "match_distance_threshold" : 0.65,
    "reid_distance_threshold" : 0.65,
    "max_stored_features" : 100 ,
    "max_idle_age" : 5,
    "reid_weight" : 0.95,
    "iou_weight" : 0.05
}

#################
# Camera Settup #
#################

# CamID is correspond to the index in camera_ip list. Ex: CamID = 0, its ip address is camera_ip[0]
ipcam_credential = ''
camera_ip = ['169.254.9.218', '169.254.10.208', '169.254.9.24', '169.254.9.234']
mqtt_broker_ip = '127.0.0.1' # '192.168.68.57'
room = 'A'

##########################
# Visualizaton in OpenCV #
##########################

debug = True # If True, send image frame for visulization, else, only send reid features
screen_size = 700