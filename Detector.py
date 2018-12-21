import cv2
import numpy as np
import os
MIN_MATCH_COUNT=30

detector=cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})


classes = ["class1.jpg", "class2.jpg", "class3.jpg", "class4.jpg", "class5.jpg"]


for image in os.listdir('test_cases'):
	print('test_cases' + image)
	cam=cv2.imread('test_cases/' + image)
	# cv2.imshow('frame', cam)
	print(image)
	count = 0
	for i in classes:
	    # ret, QueryImgBGR=cam.read()
	    # print(classes[i])
	    class_img = cv2.imread(i, 0)
	    trainKP,trainDesc=detector.detectAndCompute(class_img,None)
	    QueryImgBGR = cam
	    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
	    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
	    matches=flann.knnMatch(queryDesc,trainDesc,k=2)

	    goodMatch=[]
	    for m,n in matches:
	        if(m.distance<0.75*n.distance):
	            goodMatch.append(m)
	    if(len(goodMatch)>MIN_MATCH_COUNT):
	        tp=[]
	        qp=[]
	        for m in goodMatch:
	            tp.append(trainKP[m.trainIdx].pt)
	            qp.append(queryKP[m.queryIdx].pt)
	        tp,qp=np.float32((tp,qp))
	        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
	        h,w=class_img.shape
	        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
	        queryBorder=cv2.perspectiveTransform(trainBorder,H)
	        if count == 0:
	        	color = (255,0,0)
	        elif count == 1:
	        	color = (0,255,0)	
	        elif count == 2:
	        	color = (0,0,255)
	        elif count == 3:
	        	color = (255,255,0)
	        elif count == 4:
	        	color = (0,255,255)
	        count += 1

	        	
	        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,color,5)
	    else:
	        print("Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT))

	    cv2.namedWindow(image, cv2.WINDOW_NORMAL)  
	while True:      
	    cv2.imshow(image,QueryImgBGR)
	    if cv2.waitKey(10)==ord('q'):
	        break
	# cam.release()
	cv2.destroyAllWindows()
