import cv2 as cv

frame_width = 1920
frame_height = 1080


cap = cv.VideoCapture(-1)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M','J','P','G'))
cap.set(cv.CAP_PROP_FPS, 60)
cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

boxes = [
                [1, 1], 
                [3+int((frame_width)/2 - 2), 1], 
                [1, int((frame_height)/2 - 2)], 
                [3+int((frame_width)/2 - 2), 3+int((frame_height)/2 - 2)]
            ]
            
feed_num = 1

num=0

while cap.isOpened():

    succes, img = cap.read()
    img = img[
        boxes[feed_num][1]:int(boxes[feed_num][1]+int((frame_height)/2 - 2)), 
        boxes[feed_num][0]:int(boxes[feed_num][0]+int((frame_width)/2 - 2))
    ]

    k = cv.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv.imwrite('images/img' + str(num) + '.png', img)
        print("image saved!")
        num += 1

    cv.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()