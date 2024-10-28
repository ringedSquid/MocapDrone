import cv2 as cv 
import numpy as np 
from multiprocessing import Pipe, Process
from termcolor import colored

class PointEpipolePipeline:
    def __init__(self, pipe_in, pipe_out, fMat, iMat, Dist, w=958, h=538, debug_pipe_out=None, feedNum=0, debug=False):
        self.debug = debug
        self.pipe_in = pipe_in
        self.pipe_out = pipe_out
        self.debug_pipe_out = debug_pipe_out
        self.feedNum = feedNum
        self.fMat = fMat #fundamental Matrix between left cam and right cam
        self.iMat = iMat
        self.Dist = Dist
        self.UDistmat, self.roi = cv.getOptimalNewCameraMatrix(self.iMat, self.Dist, (w, h), 1, (w, h))
        self.process = Process(target=self.run)
        self.process.start()

        if (self.debug):
            print(colored(f"Process PointEpipolePipeline {self.feedNum} started with PID {self.process.pid}", "green"))
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.process.terminate()
        self.process.join()
        if (self.debug):
            print(colored(f"Process PointEpipolePipeline {self.feedNum} with PID {self.process.pid} killed", "yellow"))
        

    def run(self):
        while True:
            img = self.pipe_in.recv()
            img = cv.undistort(img, self.iMat, self.Dist, None, self.UDistmat)
            x, y, w, h = self.roi 
            img = img[y:y+h, x:x+w]
            points = 0
            if (self.debug):
                points, img = self.findPoints(img)
            else:
                points, _ = self.findPoints(img)
            
            #lines = cv.computeCorrespondEpilines(points.reshape(-1, 1, 2), 1, self.fMat)
            lines = 0
            #if (self.debug):
            #    img = self.drawLines(img, lines)
            
            try: #if pipe is not ready, skip!
                self.pipe_out.send((points, lines))
            except:
                pass

            if (self.debug):
                try: #if pipe is not ready, skip!
                    self.debug_pipe_out.send(img)
                except:
                    pass

    def distPointLine(self, point, line):
        return abs(line[0]*point[0] + line[1]*point[1] + line[2])/np.sqrt(line[0]**2 + line[1]**2)

    def drawLines(self, img, lines):
        rows, cols = img.shape
        for l in lines:
            c = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1] ])
            x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img = cv.line(img, (x0, y0), (x1, y1), c, 1)
        return img

    def findPoints(self, img):
        points = []
        buff = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        buff = cv.threshold(buff, 255*0.7, 255, cv.THRESH_BINARY)[1]
        contours, hierarchy = cv.findContours(buff, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for c in contours:
            moments = cv.moments(c)
            p = 0
            if (cv.contourArea(c) > 0):
                p = (
                    int(moments["m10"] / moments["m00"]),
                    int(moments["m01"] / moments["m00"])
                )
            else:
                c = c[0][0].tolist()
                p = (c[0], c[1])
            
            if (type(p) != int):
                if (self.debug):
                    cv.circle(img, (p[0], p[1]), 1, (0, 0, 255), -1)
                    cv.putText(
                        img, 
                        f"({p[0]}, {p[1]}", 
                        (p[0], p[1]-10),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 255), 1
                    )

                points.append(p)
        
        points = np.int32(points)
        return points, img

if __name__ == "__main__":
    from VideoSplitter import VideoSplitter
    from VideoViewer import VideoViewer
    from DebugTools import NullInput
    import json

    DATA_PATH = "../../data/"

    #Pipes
    splitter_out = [0] * 4
    pipeline_in = [0] * 4
    pipeline_out = [0] * 4
    null_in = [0]*4
    pipeline_debug_out = [0] * 4
    viewer_in = [0] * 4

    #pipeline processes
    pipelines = [0] * 4

    for i in range(4):
        splitter_out[i], pipeline_in[i] = Pipe()
        pipeline_out[i], null_in[i] = Pipe()
        pipeline_debug_out[i], viewer_in[i] = Pipe()

        with open(f"{DATA_PATH}InternalParamsA.json", "r") as file:
            data = json.load(file)

        pipelines[i] = PointEpipolePipeline(
            pipeline_in[i], pipeline_out[i], 
            0, np.array(data["Mint"]), np.array(data["Dist"]), 
            debug_pipe_out=pipeline_debug_out[i], feedNum=0, debug=True
        )
    
    null = NullInput(null_in, debug=False)
    vs = VideoSplitter(-1, splitter_out, debug=True)
    vout = VideoViewer(viewer_in, debug=True)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        null.__exit__(None, None, None)
        vs.__exit__(None, None, None)
        vout.__exit__(None, None, None)
        for p in pipelines:
            p.__exit__(None, None, None)
