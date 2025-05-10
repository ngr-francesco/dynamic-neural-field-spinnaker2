import matplotlib.pyplot as plt
import numpy as np

out_spikes = np.load("data/out_spikes.npz")
times = out_spikes["times"]
indices = out_spikes["indices"]
in_spikes = np.load("data/input_spikes.npz")
in_indices = in_spikes["indices"]
in_times = in_spikes["times"]
plt.plot(times, indices, "|", ms=2)
plt.plot(in_indices, in_times, '|', ms=2, alpha=0.5)
plt.savefig("plots/dnf_raster.png")
plt.show()