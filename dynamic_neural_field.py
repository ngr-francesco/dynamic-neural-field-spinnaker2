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

S2IP = os.environ.get("S2_IP", "192.168.2.33")
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


def gaussian_kernel(size, sigma_exc, sigma_inh, weight_exc=1.0, weight_inh=0.5):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    dist_sq = xx**2 + yy**2

    gauss_exc = weight_exc * np.exp(-dist_sq / (2 * sigma_exc**2))
    gauss_inh = weight_inh * np.exp(-dist_sq / (2 * sigma_inh**2))
    
    kernel = gauss_exc - 0.4
    return kernel

W, H = 20, 20
field_size = (W, H)  # e.g., 50x50 grid
u = np.zeros(field_size)  # initial activity

kernel_size = 3
kernel = gaussian_kernel(size=kernel_size*2 + 1, sigma_exc=1.5, sigma_inh=2.5)

figure = plt.figure()
plt.imshow(kernel)
plt.savefig("dnf-kernel.png")
plt.show()

weights = np.zeros((W*H,W*H))

for idx in range(W*H):
    print(f"{idx+1}/{W*H}")
    for k in range(-kernel_size,kernel_size+1):
        for j in range(-kernel_size,kernel_size+1):
            target_idx = idx+j+k*W
            if target_idx >= W*H:
                target_idx = target_idx - W*H
            
            weights[idx, target_idx] = np.round(kernel[j+kernel_size,k+kernel_size]/0.5*6)
            # print(idx,target_idx, weights[idx,target_idx])

figure = plt.figure()
plt.imshow(weights, interpolation='none')
plt.colorbar(label='Connection strength')
plt.savefig("dnf.png")
plt.show()

timesteps = 400
# # # create stimulus population with 2 spike sources
input_spikes = {k: range(k,min(k+k,timesteps)) for k in range(timesteps-1)}

stim = snn.Population(size=H*W, neuron_model="spike_list", params=input_spikes, name="stim")
stim.set_max_atoms_per_core(100)

# # create LIF population with 1 neuron
neuron_params = {
    "threshold": 5.0,
    "alpha_decay": 0.9,
    "i_offset": 0.0,
    "v_init": 0.0,
    "v_reset": 0.0,  # only used for reset="reset_to_v_reset"
    "reset": "reset_to_v_reset",  # "reset_by_subtraction" or "reset_to_v_reset"
}

inh_params = {
    "threshold": 2.0,
    "alpha_decay": 1,
    "i_offset": 3.0,
    "v_init": 0.0,
    "v_reset": 0.0,
    "reset": "reset_to_v_reset"
}

pop1 = snn.Population(size=H*W, neuron_model="lif", params=neuron_params, name="pop1", record=["spikes"])
global_inh = snn.Population(size=1, neuron_model="lif", params=inh_params, name="global_inh", record=["spikes"])

# Make an approximate estimate of the max neurons per core you can have with this connectivity
neurons_per_core = int(30)
pop1.set_max_atoms_per_core(neurons_per_core)
print(f"Setting max neurons per core to {neurons_per_core}")


# # create connection between stimulus neurons and LIF neuron
# # each connection has 4 entries: [pre_index, post_index, weight, delay]
# # for connections to a `lif` population:
# #  - weight: integer in range [-15, 15]
# #  - delay: integer in range [0, 7]. Actual delay on the hardware is: delay+1

connections = sparse_connection_list_from_dense_weights(weights, 0)
print(connections[0])

input_conns = snn.Projection(pre=stim, post=pop1, connections = [[k,k,5.0,0] for k in range(H*W)])
proj = snn.Projection(pre=pop1, post=pop1, connections=connections)

inh_conns = snn.Projection(pre=global_inh, post=pop1, connections = [[0,k,-4.0,0] for k in range(H*W)])

# # create a network and add population and projections
net = snn.Network("my network")
net.add(stim, pop1, proj, input_conns, global_inh, inh_conns)

# select hardware and run network
hw = hardware.SpiNNaker2Chip(experiment_backend_type=ExperimentBackendType(EXPERIMENT_BACKEND_TYPE), 
                                    eth_ip=S2IP)
hw.run(net, timesteps)

# # get results and plot
spike_times = pop1.get_spikes()


# Raster plot variables
indices, times = helpers.spike_times_dict_to_arrays(spike_times)
# np.savez("out_spikes.npz", indices=indices, times=times)
with open("indices_times","w+") as file:
    for idx, t in zip(indices,times):
        file.write(f"{idx}, {t}\n")
tmp = []
input_spikes_indices = np.array([tmp.extend(idx for k in range(len(spikes))) for idx,spikes in input_spikes.items()]).flatten()
tmp = []
input_spikes_times = np.array(tmp.extend(list(input_spikes.values()))).flatten()
print(input_spikes_times)
# np.savez("input_spikes.npz", indices=input_spikes_indices,times=input_spikes_times)
with open("input_spikes","w+") as file:
    for idx, t in zip(input_spikes_indices,input_spikes_times):
        file.write(f"{idx}, {t}\n")

if plot_raster:
    figure = plt.figure()
    plt.plot(times, indices, "|", ms=2)
    plt.plot(input_spikes_indices, input_spikes_times, '|', ms=2, alpha=0.5)
    plt.savefig("dnf_raster.png")
    plt.show()

# Generate frames for the 2d animation
frames = spike_frame_accum(5,W*H, indices, times, timesteps)
frames = frames.reshape((frames.shape[0],W,H))
frame_idx = 0
np.save("frames.npy", frames)
if plot_activity:
    print("Plotting animation... (Ctrl+C to stop)")
    figure = plt.figure()
    while True:
        plt.imshow(frames[frame_idx])
        plt.savefig("dnf_activity.png")
        plt.show()
        print(frames[frame_idx])
        frame_idx += 1
        if frame_idx >= frames.shape[0]:
            frame_idx = 0
        time.sleep(0.5)
