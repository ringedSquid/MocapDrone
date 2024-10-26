from multiprocessing import Pipe, Process
from termcolor import colored

class NullInput:
    def __init__(self, pipes_in, debug=False):
        self.debug = debug
        self.pipes_in = pipes_in
        self.process = Process(target = self.run)
        self.process.start()
        if (self.debug):
            print(colored(f"Process NullInput with {len(self.pipes_in)} inputs started with PID {self.process.pid}", "green"))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process.terminate()
        self.process.join()
        if (self.debug):
            print(colored(f"Process NullInput with {len(self.pipes_in)} inputs and PID {self.process.pid} killed", "yellow"))

    def run(self):
        while True:
            for i in range(len(self.pipes_in)):
                data = self.pipes_in[i].recv()
                if (self.debug):
                    print(colored(f"Packet of type {type(data)} received. pid: {self.process.pid}", "blue"))


if __name__ == "__main__":
    pass


