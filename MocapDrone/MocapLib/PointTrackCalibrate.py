import numpy as np
import cv2 as cv
from scipy import linalg, optimize  # Import optimize for least squares
from multiprocessing import Process, Pipe
from termcolor import colored
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PointTrackCalibrate:
    def __init__(self, pipesIn, iMats, debug=False):
        self.debug = debug
        if len(pipesIn) != 4:
            print(colored(f"Amount of pipes {len(pipesIn)} does not match number of cameras 4", "red"))
            self.exit()

        self.pipesIn = pipesIn
        if len(iMats) != 4:
            print(colored(f"Amount of iMats {len(iMats)} does not match number of cameras 4", "red"))
            self.exit()
        
        self.iMats = iMats
        self.points = [[] for _ in range(4)]  # Initialize as lists
        self.process = Process(target=self.run)
        self.process.start()
        if self.debug:
            print(colored(f"Process PointTrackCalibrate started with PID {self.process.pid}", "green"))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.calibrate()
        self.process.join()

    def run(self):
        while True:
            temp = []
            for i in range(4):
                pts, _ = self.pipesIn[i].recv()
                if len(pts) == 1:
                    temp.append(pts[0])

            if len(temp) == 4:
                for i in range(4):
                    self.points[i].append(temp[i])

    def calibrate(self):
        if len(self.points[0]) >= 8:
            camPoses = []
            for i in range(4):
                r = (i + 1) % 4
                pL = np.array(self.points[i])
                pR = np.array(self.points[r])

                F, _ = cv.findFundamentalMat(pL, pR, cv.FM_RANSAC, 1, 0.99999)
                E = cv.sfm.essentialFromFundamental(F, self.iMats[i], self.iMats[r])

                R, t = cv.sfm.motionFromEssential(E)
                camPoses.append({"R": R, "t": t})
            
            imagePoints = [np.array(self.points[i]) for i in range(4)]
            objectPoints = self.triangulatePoints(imagePoints, camPoses)
            if objectPoints is None or len(objectPoints) == 0:
                print(colored("No object points were triangulated.", "red"))
                return

            self.bundleAdjustment(objectPoints, imagePoints, camPoses)
            
            for i, camPose in enumerate(camPoses):
                with open(f'camera_pose_{i + 1}.json', 'w') as f:
                    json.dump(camPose, f)

            # Visualize the results and block until the window is closed
            self.visualize(camPoses, objectPoints)
        else:
            print(colored(f"Not enough points for RANSAC (n >= 8) {self.process.pid}", "red"))

    def visualize(self, camPoses, objectPoints):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot camera positions
        for camPose in camPoses:
            R = camPose["R"]
            t = camPose["t"].flatten()
            ax.scatter(t[0], t[1], t[2], c='r', marker='o')  # Camera position
            # Draw camera direction (for visualization)
            ax.quiver(t[0], t[1], t[2], -R[0, 2], -R[1, 2], -R[2, 2], length=0.1, color='b')

        # Plot object points
        if objectPoints is not None and len(objectPoints) > 0:
            ax.scatter(objectPoints[:, 0], objectPoints[:, 1], objectPoints[:, 2], c='g', marker='^')  # Object points

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Camera Positions and Object Points')

        plt.show()  # This will block until the window is closed

    def triangulatePoint(self, imagePoints, cameraPoses):
        imagePoints = np.array(imagePoints)

        if len(imagePoints) <= 1:
            return [None, None, None]

        Ps = []  # Projection matrices

        for i, cameraPose in enumerate(cameraPoses):
            RT = np.c_[cameraPose["R"], cameraPose["t"]]
            P = self.iMats[i] @ RT
            Ps.append(P)

        def DLT(Ps, imagePoints):
            A = []
            for P, imagePoint in zip(Ps, imagePoints):
                A.append(imagePoint[1] * P[2, :] - P[1, :])
                A.append(P[0, :] - imagePoint[0] * P[2, :])
            
            A = np.array(A).reshape((len(Ps) * 2, 4))
            B = A.transpose() @ A
            U, s, Vh = linalg.svd(B, full_matrices=False)
            objectPoint = Vh[3, 0:3] / Vh[3, 3]

            return objectPoint

        objectPoint = DLT(Ps, imagePoints)

        return objectPoint

    def triangulatePoints(self, imagePoints, cameraPoses):
        objectPoints = []
        for imagePointsI in imagePoints:
            objectPoint = self.triangulatePoint(imagePointsI, cameraPoses)
            objectPoints.append(objectPoint)
        
        return np.array(objectPoints)

    def reprojectionError(self, imagePoints, objectPoints, cameraPoses):
        errors = []
        for imagePointsI, objectPoint in zip(imagePoints, objectPoints):
            error = self.calculateReprojectionError(imagePointsI, objectPoint, cameraPoses)
            errors.append(error)

        return np.array(errors)

    def calculateReprojectionError(self, imagePoints, objectPoint, cameraPoses):
        imagePoints = np.array(imagePoints)

        if len(imagePoints) <= 1:
            return None

        imagePointsT = imagePoints.transpose((0, 1))
        errors = []

        for i, cameraPose in enumerate(cameraPoses):
            projectedImgPoints, _ = cv.projectPoints(
                np.expand_dims(objectPoint, axis=0).astype(np.float32),
                cameraPose["R"].astype(np.float64),
                cameraPose["t"].astype(np.float64),
                self.iMats[i].astype(np.float64),
                np.array([])
            )
            projectedImgPoint = projectedImgPoints[:, 0, :][0]
            errors.append((imagePointsT[i] - projectedImgPoint).flatten() ** 2)

        return np.mean(errors)

    def bundleAdjustment(self, objectPoints, imagePoints, cameraPoses):
        numCameras = len(cameraPoses)
        numPoints = len(objectPoints)
        
        initialParams = np.concatenate([
            np.concatenate([cameraPoses[i]["R"].flatten(), cameraPoses[i]["t"].flatten()]) for i in range(numCameras)
        ] + [objectPoints.flatten()])

        def residuals(params):
            residuals = []
            for i in range(numCameras):
                R = params[i * 6:i * 6 + 3].reshape((3, 3))
                t = params[i * 6 + 3:i * 6 + 6]
                for j in range(numPoints):
                    projectedPoint, _ = cv.projectPoints(
                        objectPoints[j].reshape(1, 3),
                        R,
                        t,
                        self.iMats[i],
                        np.array([]),
                    )
                    error = imagePoints[i][j] - projectedPoint.flatten()
                    residuals.append(error)

            return np.array(residuals).flatten()

        result = optimize.least_squares(residuals, initialParams, method='lm')
        
        optimizedParams = result.x
        for i in range(numCameras):
            cameraPoses[i]["R"] = optimizedParams[i * 6:i * 6 + 3].reshape((3, 3))
            cameraPoses[i]["t"] = optimizedParams[i * 6 + 3:i * 6 + 6]

        objectPoints[:] = optimizedParams[numCameras * 6:].reshape(numPoints, 3)

if __name__ == "__main__":
    from VideoSplitter import VideoSplitter
    from VideoViewer import VideoViewer
    from PointEpipolePipeline import PointEpipolePipeline
    from DebugTools import NullInput
    import json

    DATA_PATH = "../../data/"

    #Pipes
    splitter_out = [0] * 4
    pipeline_in = [0] * 4
    pipeline_out = [0] * 4
    calibration_in = [0]*4
    pipeline_debug_out = [0] * 4
    viewer_in = [0] * 4

    #pipeline processes
    pipelines = [0] * 4
    imats = []

    for i in range(4):
        splitter_out[i], pipeline_in[i] = Pipe()
        pipeline_out[i], calibration_in[i] = Pipe()
        pipeline_debug_out[i], viewer_in[i] = Pipe()

        with open(f"{DATA_PATH}InternalParams{i}.json", "r") as file:
            data = json.load(file)
            imats.append(data)

        pipelines[i] = PointEpipolePipeline(
            pipeline_in[i], pipeline_out[i], 
            0, np.array(data["Mint"]), np.array(data["Dist"]), 
            debug_pipe_out=pipeline_debug_out[i], feedNum=0, debug=True
        )
    
    
    calibration = PointTrackCalibrate(calibration_in, imats, debug=True)
    vs = VideoSplitter(-1, splitter_out, debug=True)
    vout = VideoViewer(viewer_in, debug=True)

    try:
        while True:
            pass
    except KeyboardInterrupt:
        vs.__exit__(None, None, None)
        vout.__exit__(None, None, None)
        for p in pipelines:
            p.__exit__(None, None, None)
        calibration.__exit__(None, None, None)