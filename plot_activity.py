import matplotlib.pyplot as plt
import numpy as np
import time

frames = np.load("frames.npy")
figure = plt.figure()
frame_idx = 0
while True:
    plt.imshow(frames[frame_idx])
    plt.savefig("dnf_activity.png")
    plt.show()
    print(frames[frame_idx])
    frame_idx += 1
    if frame_idx >= frames.shape[0]:
        frame_idx = 0
    time.sleep(0.05)