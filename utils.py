import math

class ProgressBar():
    def __init__(self, steps = 20, length=20, end_with_newline=False):
        self.total_steps = steps-1
        self.length = length
        self.empty_bar = length
        self.end_with_newline = end_with_newline


    def print(self, step):
        print("\r[", end="")
        bar_len = math.ceil(((step+1)/self.total_steps) * self.length)
        bar_len = int(bar_len)
        if bar_len > self.length:
            bar_len = self.length
        empty_bar = self.length - bar_len
        for _ in range(bar_len):
            print("=", end="")
        for _ in range(empty_bar):
            print(".", end="")
        print("]", end="")
        if step == self.total_steps and self.end_with_newline: 
            print()
