import cv2 as cv 
import numpy as np 
from multiprocessing import Pipe, Process
from termcolor import colored
import time 

class VideoViewer:
    def __init__(self, pipes_in, debug=False):
        self.pipes_in = pipes_in
        if (len(pipes_in) != 4):
            print(colored(f"Amount of pipes {len(pipes)} does not match number of cameras 4", "red"))
            self.exit()

        self.debug = debug
        self.process = Process(target = self.run)
        self.process.start()
        if (debug):
            print(colored(f"Process VideoViewer started with PID {self.process.pid}", "green"))

    
    def run(self):
        t_old = time.time()
        while True:
            img = 0
            for i in range(4):
                img_in = self.pipes_in[i].recv()
                width = int(img_in.shape[1]*0.75)
                height = int(img_in.shape[0]*0.75)
                img_in = cv.resize(img_in, (width, height))
                if (type(img) == int):
                    img = img_in
                else:
                    img = cv.vconcat([img, img_in])
            
            if (type(img) != int):
                #print(img.shape)
                if (self.debug):
                    fps = str(int(1/(time.time()-t_old))) 
                    cv.putText(img, fps, (5, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0)) 

                cv.imshow(f"Video Out", img)
                if cv.waitKey(1) == 27:
                    break
                
                t_old = time.time()
        
        self.exit()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process.terminate()
        self.process.join()
        if (self.debug):
            print(colored(f"Process VideoViewer with PID {self.process.pid} killed", "yellow"))

if __name__ == "__main__":
    pass


