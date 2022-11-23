import base64
from torchreid.utils import FeatureExtractor
import time
import cv2
import numpy as np
import config as cfg
import torch
import json
import config as cfg
from threading import Thread
import paho.mqtt.client as mqtt
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, \
    scale_coords,  set_logging, jsonEncoder
from utils.datasets import letterbox
from utils.torch_utils import select_device


class vStream:
    def __init__(self,src,width,height):
        self.width=width
        self.height=height
        self.capture=cv2.VideoCapture(src)
        self.thread=Thread(target=self.update,args=())
        self.thread.daemon=True
        self.thread.start()
    def update(self):
        while True:
            _,self.frame=self.capture.read()
            self.frame2=cv2.resize(self.frame,(self.width,self.height))
    def getFrame(self):
        return self.frame2

if __name__ == "__main__":

    mqttBroker = cfg.mqtt_broker_ip
    client = mqtt.Client(f"room{cfg.room}")
    client.connect(mqttBroker)  
    # Initialize
    set_logging()
    device = select_device('0') 
    # Load Yolov7 model
    model = attempt_load(cfg.yolo_weight_path, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(cfg.input_size, s=stride)  # check image size
    model.half()  # to FP16
    cudnn.benchmark = True  
    # Prepare GPU inference
    if device.type != 'cpu':
        output, out = model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once for warming up     


    # Instantiate ReID Network osnet_x1_0 or osnet_ain_x1_0
    extractor = FeatureExtractor(model_name=cfg.reid_weight_path[7:-4],
                                 model_path=cfg.reid_weight_path,
                                device='cuda')  

    # Create a VideoCapture object and read from input file
    cam = [None for _ in range(len(cfg.camera_ip))]  
    for i in range(len(cfg.camera_ip)):
        cam[i] = vStream(f"rtsp://{cfg.ipcam_credential}@{cfg.camera_ip[i]}:554/live1s1.sdp", 1280, 720)   
    n_del = 0
    tot_del = 0 

    # Start video inferencing (Make sure each thread has been executed )
    haha = input("Start streaming ?")
    while True:
        # Capture frame-by-frame  
        frame = [None]*len(cam)
        for i in range(len(frame)):
            frame[i] = cam[i].getFrame()    
        start = time.time()
        imgs = frame.copy()   

        # Padded resize
        for i in range(len(imgs)):
            imgs[i] = letterbox(imgs[i], cfg.input_size, stride=stride)[0]

        # Convert
            imgs[i] = imgs[i][:, :, ::-1].transpose(2, 0, 1)
            imgs[i] = np.ascontiguousarray(imgs[i])
            imgs[i] = torch.from_numpy(imgs[i]).to(device)
            imgs[i] = imgs[i].half()  # uint8 to fp16/32
            imgs[i] /= 255.0  # 0 - 255 to 0.0 - 1.0    
        img_input = torch.cat([imgs[i].unsqueeze(0) for i in range(len(imgs))], dim=0)    
        yolo_time = time.time()

        # Yolov7 Inference 
        pred = model(img_input, augment=False)[0] 
        print(f'Yolo inference time: {time.time()-yolo_time}')

        # Apply NMS   
        preds = [[] for _ in range(len(imgs))]
        reid_imgs = []
        reid_xyxy = [[] for _ in range(len(imgs))]
        reid_feat = [[] for _ in range(len(imgs))]
        num_person = [0 for _ in range(len(imgs))]
        b_frame = [[] for _ in range(len(imgs))]  
        nms_time = time.time()    
        preds = non_max_suppression(pred,
                                    cfg.yolo["conf_thres"],
                                    cfg.yolo["iou_thres"], [0],
                                    cfg.yolo["agnostic_nms"],
                                    )

        # Draw rectangles and labels on the orig inal image
        for i in range(len(preds)):
            det = preds[i]
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_input[i].unsqueeze(0).shape[2:], det[:, :4],
                                      frame[i].shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1 = int(xyxy[0])
                x2 = int(xyxy[2])
                y1 = int(xyxy[1])
                y2 = int(xyxy[3])   
                currarea = (x2 - x1) * (y2 - y1)
                if (currarea > 500):
                    im_input = frame[i][y1:y2, x1:x2]
                    cvt_img = im_input[:, :, ::-1]  # to RGB
                    reid_imgs.append(cvt_img)
                    num_person[i] += 1
                    reid_xyxy[i].append([x1, y1, x2, y2])   
        print(f'NMS inference time: {(time.time()-nms_time)}')

        # ReID Inference
        reid_time = time.time()
        if (len(reid_imgs) != 0):
            reid_feats = extractor(reid_imgs)
            reid_feats = reid_feats.to('cpu')   
            for i in range(len(imgs)):
                if i == 0:
                    curr_num = num_person[i]
                    reid_feat[i] = reid_feats[:curr_num]
                else:
                    reid_feat[i] = reid_feats[curr_num:curr_num+num_person[i]]
                    curr_num += num_person[i] 
        print(f'Reid inference time: {time.time()-reid_time}')    

        # Publish payload data to server 
        #--------------------MQtt--------------------
        mqtt_time = time.time()
        if cfg.debug:
            for i in range(len(imgs)):
                b_frame[i] = base64.b64encode(frame[i]).decode('ascii')
            send_message = [{'Frame': b_frame[i], 'Feat': reid_feat[i], 'CamID': i, "XyXy": reid_xyxy[i]} for i in range(len(imgs))]    
        else: 
            send_message = [{'Feat': reid_feat[i], 'CamID': i, "XyXy": reid_xyxy[i]} for i in range(len(imgs))] 

        send_json = json.dumps(send_message, cls=jsonEncoder)
        client.publish(f"room{cfg.room}", send_json)
        print(f"Publish to mqtt duration: {time.time()-mqtt_time}")
        
        end = time.time()
        n_del += 1 
        timeDiff = end - start
        tot_del += timeDiff
        print(f"Overall FPS: {1/timeDiff}")
        # if (timeDiff < 1.0/(15)):
        #   time.sleep(1.0/(15) - timeDiff)   