import zmq
import cv2
import json
import threading
import time
import base64
import random
import numpy as np
import datetime
from paho.mqtt.client import Client
from utils.general import xyxy2bc
from utils.multicamera import TrackDatabase, PersonDatabase
from utils.plots import plot_one_box
import config as cfg

camera_db = [0 for _ in range(len(cfg.camera_ip))]

def on_message(client, userdata, msg):
	global camera_db
	print("Converting from Json to Object")
	m_in=json.loads(msg.payload) #decode json data
	camera_db = m_in	
	if cfg.debug:
		for i in range(len(camera_db)):
			camera_db[i]["Frame"] = np.frombuffer(base64.b64decode(camera_db[i]["Frame"].encode('ascii')), dtype=np.uint8)
			camera_db[i]["Frame"] = camera_db[i]["Frame"].reshape((720, 1280, 3))

def on_connect(client, userdata, flags, rc):
	print("Connected with result code "+str(rc))

def recvCameraInfo():
	global camera_db
	client = Client("Server")
	client.connect("127.0.0.1")
	client.subscribe("roomA")
	client.on_connect = on_connect
	client.on_message = on_message
	client.loop_start()

def reidMatching():
	global camera_db
	tracks = TrackDatabase(len(cfg.camera_ip), cfg.matching["max_idle_age"])
	persons = PersonDatabase(cfg.matching["max_stored_features"],
	                         cfg.matching["match_distance_threshold"],
	                         cfg.matching["iou_weight"],
	                         cfg.matching["reid_weight"],
	                         cfg.matching["reid_distance_threshold"])
	n_del = 0
	tot_del = 0
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]
	person_json = []
	time.sleep(1.0)
	while True:
	
		camera_db_temp = camera_db.copy()  # Create deep copy of global variable
		for camid in range(len(cfg.camera_ip)):
			if camera_db_temp[camid]:
				start = time.time()
				curr_cam = camera_db_temp[camid]
				match_arr = persons.feature_matching(curr_cam["Feat"], tracks.get_iou_matrix(camid, curr_cam["XyXy"], persons.get_num_person()))
				curr_tracks = tracks.update_tracks(camid, curr_cam["XyXy"], match_arr)
				# print(curr_cam)
				for key, value in curr_tracks.items():
					xyxy = [int(value.xyxy[0]), int(value.xyxy[1]), int(value.xyxy[2]), int(value.xyxy[3])]
					feat = persons._db[value.personid]._feats[-1]
					realx = 5 / 1280
					realy = 5 / 720
					bc = xyxy2bc(xyxy)
					person_json.append({"camid": camid, "person_id": str(value.personid), "xyxy": xyxy, 'bc': bc,"time": str(datetime.datetime.now())})
					if cfg.debug:
						plot_one_box(value.xyxy,
									camera_db_temp[camid]["Frame"],
									label=str(value.personid),
									color=colors[int(value.personid)],
									line_thickness=5)

				jsonString = json.dumps(person_json)
				with open('json_data.json', 'w', encoding='utf-8') as outfile:
					outfile.write(jsonString)
				if cfg.debug:
					cv2.namedWindow("Cam" + str(camid), cv2.WINDOW_NORMAL)
					cv2.resizeWindow("Cam" + str(camid), cfg.screen_size, cfg.screen_size)
					cv2.imshow("Cam" + str(camid), camera_db_temp[camid]["Frame"])
				end = time.time()
				n_del += 1
				tot_del += end - start
				if not tot_del == 0:
					print(n_del / tot_del)
			if cv2.waitKey(1) == 27:
				break


p1 = threading.Thread(target=recvCameraInfo, args=())
p1.start()

p2 = threading.Thread(target=reidMatching, args=())
p2.start()
