import cv2 as cv 
import numpy as np 
from multiprocessing import Pipe, Process
from termcolor import colored

class VideoSplitter:
    def __init__(self, cameraIndex, pipes, frame_width=1920, frame_height=1080, debug=False):
        self.debug = debug
        self.cap = cv.VideoCapture(cameraIndex)
        self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M','J','P','G'))
        self.cap.set(cv.CAP_PROP_FPS, 60)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.box_width = int((frame_width)/2 - 2)
        self.box_height = int((frame_height)/2 - 2)
        self.pipes = pipes
        if (len(pipes) != 4):
            print(colored(f"Amount of pipes {len(pipes)} does not match number of cameras 4", "red"))
            self.exit()
        
        self.process = Process(target=self.run)
        self.process.start()
        if (self.debug):
            print(colored(f"Process VideoSplitter started with PID {self.process.pid}", "green"))

    def __exit__(self, exc_type, exc_value, traceback):
        self.process.terminate()
        self.process.join()
        self.cap.release()
        if (self.debug):
            print(colored(f"Process VideoSplitter with PID {self.process.pid} killed", "yellow"))

    def run(self):
        while True:
            retval, frame = self.cap.read()
            boxes = [
                [1, 1], 
                [3+self.box_width, 1], 
                [1, 3+self.box_height], 
                [3+self.box_width, 3+self.box_height]
            ]
            for i in range(4):
                try: #if pipe is not ready, skip!
                    self.pipes[i].send(
                        frame[
                            boxes[i][1]:int(boxes[i][1]+self.box_height), 
                            boxes[i][0]:int(boxes[i][0]+self.box_width)
                        ].copy()
                    )
                except:
                    pass

            
            
                

if __name__ == "__main__":
    from VideoViewer import VideoViewer
    splitter_out = []
    viewer_in = []
    for i in range(4):
        parent_conn, child_conn = Pipe()
        splitter_out.append(parent_conn)
        viewer_in.append(child_conn)

    vs = VideoSplitter(-1, splitter_out, debug=True)
    vout = VideoViewer(viewer_in, debug=True)

    while True:
        message = input()
        if (message == "stop"):
            vs.__exit__(None, None, None)
            vout.__exit__(None, None, None)



    

