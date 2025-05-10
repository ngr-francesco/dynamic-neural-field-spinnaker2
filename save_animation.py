import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load frames
frames = np.load("frames.npy")  # shape: (num_frames, H, W)
assert frames.ndim == 3, "frames.npy must have shape (frames, height, width)"

# Consistent color scaling
vmin, vmax = np.min(frames), np.max(frames)

# Set up figure
fig, ax = plt.subplots()
img = ax.imshow(frames[0], cmap='viridis', vmin=vmin, vmax=vmax, animated=True)
ax.axis("off")  # Remove axes

# Update function
def update(i):
    img.set_array(frames[i])
    return [img]

# Animate
ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=20, blit=True)

# Save GIF
ani.save("dnf_activity_fixed_spike_size.gif", writer="pillow", fps=60)
print("Saved GIF to dnf_activity.gif")
