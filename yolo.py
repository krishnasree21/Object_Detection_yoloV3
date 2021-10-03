import cv2
import numpy as np


cap = cv2.VideoCapture(0)
whT=320
confThreshold = 0.5
nmsThreshold = 0.3

classesFile='coco.names'
classNames = []

with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)
modelConfi = 'yolov3-320.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfi,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT,wT,cT = img.shape
    boundingbox = []
    classIds = []
    confidencevalue = []

    for output in outputs:
        for detect in output:
            scores = detect[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(detect[2]*wT) , int(detect[3]*hT)
                x,y = int((detect[0]*wT)-w/2) , int((detect[1]*hT)-h/2)
                boundingbox.append([x,y,w,h])
                classIds.append(classId)
                confidencevalue.append(float(confidence))
    #print(len(boundingbox))
    indices = cv2.dnn.NMSBoxes(boundingbox,confidencevalue,confThreshold,nmsThreshold)
    #print(indices)
    for i in indices:
        i = i[0]
        box = boundingbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confidencevalue[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)


while True:
    success, img=cap.read()

    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    #print(layerNames)
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
   # print(outputNames)
    #print(net.getUnconnectedOutLayers())

    outputs = net.forward(outputNames)
    #print(outputs[0].shape)
    findObjects(outputs,img)



    cv2.imshow('Image',img)
    cv2.waitKey(1)

