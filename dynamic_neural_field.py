"""
Example of Leaky Integrate&Fire neuron where synaptic events directly increase
the membrane voltage.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from spinnaker2 import hardware, helpers, snn
from spinnaker2.experiment_backends.base_experiment_backend import ExperimentBackendType
from spinnaker2.experiment_backends.backend_settings import BackendSettings, ROUTING
from spinnaker2.ann2snn_helpers import sparse_connection_list_from_dense_weights
import time

import logging

logging.basicConfig(level=logging.INFO)

S2IP = os.environ.get("S2_IP", "192.168.2.52")
STM_IP = os.environ.get("STM_IP48", "192.168.4.2")

EXPERIMENT_BACKEND_TYPE = os.environ.get("PYS2_EXPERIMENT_BACKEND", "spinnman2")

settings = BackendSettings()

settings.routing_type = ROUTING.MC

plot_activity = False

plot_raster = True

import numpy as np

def spike_frame_accum(timesteps_per_frame, n_neurons, indices, times, t_max):
    n_frames = int(np.ceil(t_max / timesteps_per_frame))
    frames = np.zeros((n_frames, n_neurons))
    for idx, t in zip(indices, times):
        frames[int(np.floor(t/timesteps_per_frame)), idx] += 1
    return frames


def gaussian_kernel(size, sigma_exc, weight_exc=1.0):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    dist_sq = xx**2 + yy**2

    gauss_exc = weight_exc * np.exp(-dist_sq / (2 * sigma_exc**2))
    
    kernel = gauss_exc - 0.4
    return kernel

# Size of the neural field
W, H = 20, 20
field_size = (W, H)  # e.g., 50x50 grid
u = np.zeros(field_size)  # initial activity

kernel_size = 3
kernel = gaussian_kernel(size=kernel_size*2 + 1, sigma_exc=1.)

# Show the spatial distribution of weights in the kernel
figure = plt.figure()
plt.imshow(kernel)
plt.savefig("plots/dnf-kernel.png")
plt.show()

# Create the weight matrix according to the gaussian kernel
weights = np.zeros((W*H,W*H))
for idx in range(W*H):
    print(f"{idx+1}/{W*H}")
    for k in range(-kernel_size,kernel_size+1):
        for j in range(-kernel_size,kernel_size+1):
            target_idx = idx+j+k*W
            if target_idx >= W*H:
                target_idx = target_idx - W*H
            
            weights[idx, target_idx] = np.round(kernel[j+kernel_size,k+kernel_size]/0.5*3)
            # print(idx,target_idx, weights[idx,target_idx])

# Plot the entire weight matrix
figure = plt.figure()
plt.imshow(weights, interpolation='none')
plt.colorbar(label='Connection strength')
plt.savefig("plots/dnf.png")
plt.show()

# # # Create some input spike patterns (a bit random)
timesteps = 400
input_spikes_incremental = {k: list(range(k,min(k+k,timesteps))) for k in range(timesteps-1)}
input_spikes_fixed = {k: list(range(k,min(k+5,timesteps))) for k in range(timesteps-1)}

# Just for conveniency, you can choose which one of the above you want to use
input_spikes = input_spikes_incremental

# Instantiate the spike list that will send the input spikes to the population
stim = snn.Population(size=H*W, neuron_model="spike_list", params=input_spikes, name="stim")

# This number can be tweaked to maximize the number of neurons per core, if necessary
stim.set_max_atoms_per_core(100)


# Parameters for a LIF neuron with exponential synapses
neuron_params = {
    "threshold": 10.0,
    "alpha_decay": 0.8,
    "i_offset": 0.0,
    "v_init": 0.0,
    "v_reset": -0.2,  # only used for reset="reset_to_v_reset"
    "reset": "reset_by_subtraction",  # "reset_by_subtraction" or "reset_to_v_reset"
    "exc_decay": 0.5,
    "inh_decay": 0.2,
    "t_refrac": 0,
}

# Inhibitory neurons with a bias current that forces them to spike regularly
inh_params = {
    "threshold": 5.0,
    "alpha_decay": 1,
    "i_offset": 3.0,
    "v_init": 0.0,
    "v_reset": 0.0,
    "reset": "reset_to_v_reset"
}

# Initialize the neural field 
pop1 = snn.Population(size=H*W, neuron_model="lif_curr_exp", params=neuron_params, name="pop1", record=["spikes"])

# Initialize the global inhibitory population
global_inh = snn.Population(size=1, neuron_model="lif", params=inh_params, name="global_inh", record=["spikes"])

# Can be played with for optimizing number of cores used
neurons_per_core = int(30)
pop1.set_max_atoms_per_core(neurons_per_core)
print(f"Setting max neurons per core to {neurons_per_core}")


# # create connection between stimulus neurons and LIF neuron
# # each connection has 4 entries: [pre_index, post_index, weight, delay]
# # for connections to a `lif` population:
# #  - weight: integer in range [-15, 15]
# #  - delay: integer in range [0, 7]. Actual delay on the hardware is: delay+1


# Define the connections:
# First input connections (weight value is random, can be tuned)
input_conns = snn.Projection(pre=stim, post=pop1, connections = [[k,k,5.0,0] for k in range(H*W)])

# Recurrent connections from the weight matrix we calculated before
connections = sparse_connection_list_from_dense_weights(weights, 0)
proj = snn.Projection(pre=pop1, post=pop1, connections=connections)

# Inhibitory connections
inh_conns = snn.Projection(pre=global_inh, post=pop1, connections = [[0,k,-4.0,0] for k in range(H*W)])

# # create a network and add population and projections
net = snn.Network("my network")
net.add(stim, pop1, proj, input_conns, global_inh, inh_conns)

# select hardware and run network
hw = hardware.SpiNNaker2Chip(experiment_backend_type=ExperimentBackendType(EXPERIMENT_BACKEND_TYPE), 
                                    eth_ip=S2IP)
hw.run(net, timesteps)

# # get results and plot --------- From here on: unrelated to spinnaker
spike_times = pop1.get_spikes()


# Raster plot variables
indices, times = helpers.spike_times_dict_to_arrays(spike_times)
# np.savez("data/out_spikes.npz", indices=indices, times=times)
with open("data/indices_times","w+") as file:
    for idx, t in zip(indices,times):
        file.write(f"{idx}, {t}\n")
tmp = []
input_spikes_times = []
input_spikes_indices = []
for idx, spikes in input_spikes.items():
    input_spikes_indices.extend([idx for _ in range(len(spikes))])
    input_spikes_times.extend(spikes)
print(input_spikes_times)
# np.savez("data/input_spikes.npz", indices=input_spikes_indices,times=input_spikes_times)
with open("data/input_spikes","w+") as file:
    for idx, t in zip(input_spikes_indices,input_spikes_times):
        file.write(f"{idx}, {t}\n")

if plot_raster:
    figure = plt.figure()
    plt.plot(times, indices, "|", ms=2)
    plt.plot(input_spikes_indices, input_spikes_times, '|', ms=2, alpha=0.5)
    plt.savefig("plots/dnf_raster.png")
    plt.show()

# Generate frames for the 2d animation
frames = spike_frame_accum(5,W*H, indices, times, timesteps)
frames = frames.reshape((frames.shape[0],W,H))
frame_idx = 0
np.save("data/frames.npy", frames)
if plot_activity:
    print("Plotting animation... (Ctrl+C to stop)")
    figure = plt.figure()
    while True:
        plt.imshow(frames[frame_idx])
        plt.savefig("plot/dnf_activity.png")
        plt.show()
        print(frames[frame_idx])
        frame_idx += 1
        if frame_idx >= frames.shape[0]:
            frame_idx = 0
        time.sleep(0.5)
