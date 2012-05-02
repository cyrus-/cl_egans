# All content Copyright 2010 Cyrus Omar <cyrus.omar@gmail.com> unless otherwise
# specified.
#
# Contributors:
#     Cyrus Omar <cyrus.omar@gmail.com>
#
# This file is part of, and licensed under the terms of, the atomic-hedgehog
# package.
#
# The atomic-hedgehog package is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# The atomic-hedgehog package is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with the atomic-hedgehog package. If not, see <http://www.gnu.org/licenses/>.
"""This is a sample script used for testing. 

Implements the Brette et al, 2007 COBA benchmark. See paper for details.
"""

import numpy
import clq.backends.opencl.pyopencl as cl 
from cypy import tic, toc

# Set up an OpenCL context
ctx = cl.ctx = cl.Context.for_device(0, 0)

# Create the root node of the simulation
from ahh.cl.egans import Simulation
sim = Simulation(ctx,
    n_realizations=1,
    n_realizations_per_division_max=1,
    n_timesteps = 10000,
    DT=0.1)

# Create 4000 LIF neurons
N_Exc = 3200
N_Inh = 800
N = N_Exc + N_Inh
from ahh.cl.egans.spiking.models import ReducedLIF
from ahh.cl.egans.spiking import InitializeFromHost
neurons = ReducedLIF(sim, "LIF", 
    count=N,
    tau=20.0,
    v_reset=0.0,
    v_thresh=10.0,
    abs_refractory_period=5.0)
# initialize the voltage with a random normal vector generated on the host
InitializeFromHost(neurons.v, 
   lambda shape, dtype: numpy.random.normal(-5.0, 5.0, shape).astype(dtype)) 

# Create excitatory and inhibitory synapses
from ahh.cl.egans.spiking.inputs import ExponentialSynapse
e_synapse = ExponentialSynapse(neurons, 'ge',
    tau=5.0,
    reversal=60.0)
InitializeFromHost(e_synapse.g,
    lambda shape, dtype: numpy.random.normal(4.0, 1.5, shape).astype(dtype))

i_synapse = ExponentialSynapse(neurons, 'gi',
    tau=10.0,
    reversal=-20.0)
InitializeFromHost(i_synapse.g, 
    lambda shape, dtype: numpy.random.normal(20.0, 12.0, shape).astype(dtype))

# Insert Poisson spikes
from ahh.cl.egans.spiking.inputs import LocalPoisson
e_poisson = LocalPoisson(e_synapse, rate=100)

# Create connectivity matrix
from ahh.cl.egans import ConstantArray
from ahh.np import DirectedAdjacencyMatrix
cm = DirectedAdjacencyMatrix(N)
cm.connect_randomly(0.0)
neighbor_data = ConstantArray(sim, "neighbor_data", cm.packed)

# Set up a spike propagation algorithm
from ahh.cl.egans.spiking.connectivity import AtomicReceiver, AtomicSender
e_receiver = AtomicReceiver(e_synapse, weight=0.6)
i_receiver = AtomicReceiver(i_synapse, weight=6.7)
sender = AtomicSender(neurons,
    neighbor_data=neighbor_data,
    target_calculation="ge if idx_model < %d else gi" % N_Exc)
sender.ge = e_receiver.alloc_out
sender.gi = i_receiver.alloc_out

from ahh.cl.egans import AccumulateOnHost

# Set up a spike raster probe
#from ahh.cl.egans.spiking.probes import SpikeRasterProbe
#raster_probe = AccumulateOnHost(SpikeRasterProbe(neurons))

# Set up voltage expression probe
#from ahh.cl.egans import ExpressionProbe, AccumulateOnHost
#v_probe = AccumulateOnHost(ExpressionProbe(neurons,
#    expression="v",
#    cl_dtype=cl.cl_float,
#    hook="post_read_state"))

# Set up spike list probe
#from ahh.cl.egans.spiking.probes import SpikeListProbe
#spike_list_probe = AccumulateOnHost(SpikeListProbe(neurons))
#
## Set up binned spike count probe
#from ahh.cl.egans.spiking.probes import BinnedSpikeCountProbe
#binned_spike_count_probe = AccumulateOnHost(BinnedSpikeCountProbe(neurons,
#    bin_size=100,
#    shift_size=100))

# Set up spike scatter probe
from ahh.cl.egans.spiking.probes import SpikeScatterProbe
spike_scatter_probe = SpikeScatterProbe(neurons)

# Finalize specification
sim.finalize()

# Generate code
tic("Generating code...")
sim.generate()
toc()
print sim.code
import ast
ast.dump(ast.parse(sim.code))

# Allocate memory
tic("Allocating memory...")
sim.allocate()
toc()
sim.print_memory_summary()

print sim.constants
step_fn = sim._step_fn_even
#print step_fn.free_variables
concrete_fn = step_fn.get_concrete_fn(cl.cl_int, cl.cl_int)
print concrete_fn.program_source
#print concrete_fn.generate_kernel(ctx)
#print "IT WORKS"

# Run simulation
tic("Running...")
sim.run()
toc()
