import cv2 as cv 
import numpy as np 
from threading import Thread, Lock
import time

class VideoSplitter:
    def __init__(self, cameraIndex, frame_width=1920, frame_height=1080):
        self.cap = cv.VideoCapture(cameraIndex)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.feed_images = [0] * 4
        self.read_lock = Lock()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.box_width = int((frame_width)/2 - 2)
        self.box_height = int((frame_height)/2 - 2)
        self.started = False
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.cap.release()

    def start(self):
        if (self.started == False):
            self.started = True
            self.thread = Thread(target=self.update, args=())
            self.thread.start()
            return self

    def stop(self):
        if (self.started == True):
            self.started = False
            self.thread.join()

    def update(self):
        while self.started:
            retval, frame = self.cap.read()
            boxes = [
                [1, 1], 
                [3+self.box_width, 1], 
                [1, 3+self.box_height], 
                [3+self.box_width, 3+self.box_height]
            ]
            self.read_lock.acquire()
            for i in range(4):
                self.feed_images[i] = frame[
                    boxes[i][1]:int(boxes[i][1]+self.box_height), 
                    boxes[i][0]:int(boxes[i][0]+self.box_width)
                ]
                
            self.read_lock.release()


    def read(self, index):
        if (index > -1 and index < 4):
            self.read_lock.acquire()
            out = self.feed_images[index]
            self.read_lock.release()
            return out
        
    def bulkRead(self):
        out = [0] * 4
        self.read_lock.acquire()
        for i in range(4):
            out[i] = self.feed_images[i]
        self.read_lock.release()
        return out
    
    def isReady(self):
        self.read_lock.acquire()
        ready = (type(self.feed_images[3]) != int)
        self.read_lock.release()
        return ready

if __name__ == "__main__":
    vs = VideoSplitter(-1, 1280, 1080).start()
    while (vs.isReady() == False):
        pass

    while True:
        frame = vs.read(0)
        if (frame.any()):
            cv.imshow('webcam', frame)
            if cv.waitKey(1) == 27:
                break
    
    vs.stop()
    cv.destroyAllWindows
    

