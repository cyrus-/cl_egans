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
"""Connectivity and communication stuff lives here."""
#@PydevCodeAnalysisIgnore

import ahh.py as py
import ahh.cl as cl
import ahh.cl.ements as clements
from ahh.cl.egans import Node, Allocation

class AtomicSender(Node):
    """Sends spikes using atomic operations."""
    
    @py.autoinit
    def __init__(self, parent, basename="AtomicSender"): pass
    
    neighbor_data = None
    """The jagged matrix of neighbor data."""
    
    neighbors_calculation = staticmethod(lambda g: 
        """
        neighbors_offset = neighbor_data[idx_realization]
        neighbor_size = neighbor_data[neighbors_offset]
        neighbors = neighbor_data + neighbors_offset + 1
        """ << g)
    
    target_calculation = None
    """The calculation to use to determine which buffer to send spikes to
    for the current neuron."""
    
    i_stride = 1
    """How many elements to stride when looping over neighbor list."""
    
    int_weight = 1
    """Expression to use to calculate the integer-valued weight for the spike."""
    
    def in_spike_propagation(self, g):
        """
        target = target_calculation
        neighbors_calculation
        for i in (0, neighbor_size, i_stride):
        """ << g
        g.tab << g
        self.trigger_staged_cg_hook("spike_send", g)
        (g.untab, "\n") << g
        
    def in_spike_send(self, g):
        "atom_add(target + realization_first_idx_div + neighbors[i], int_weight)" >> g
        
    def pre_step_kernel_body(self, g):
        # TODO: remove this once extension inference works
        g << 'exec "' << cl.cl_khr_global_int32_base_atomics.pragma_str[1:-1] << '"\n'
    
class AtomicReceiver(Node):
    """Receives spikes and converts them into conductance updates. The parent
    should be the synapse."""
    @py.autoinit
    def __init__(self, parent, basename="AtomicReceiver"): pass
    
    weight = 1.0
    """Weight expression."""
    
    reader = "alloc_in[idx_state]"
    """Readout expression"""
    
    @py.lazy(property)
    def _buffer_size(self):
        return (self.model.count * self.sim.n_realizations_per_division_max,)
    
    @py.lazy(property)
    def alloc_in(self):
        return Allocation(self, "in", self._buffer_size, cl.cl_int)
    
    @py.lazy(property)
    def buffer_in(self):
        return self.alloc_in.buffer
    
    @py.lazy(property)
    def alloc_out(self):
        return Allocation(self, "out", self._buffer_size, cl.cl_int)
    
    @py.lazy(property)
    def buffer_out(self):
        return self.alloc_out.buffer
    
    def on_initialize_memory(self, timestep_info):
        clements.ew_set_0(self.buffer_in)
        clements.ew_set_0(self.buffer_out)
    
    def on_prepare_step_fn_odd(self):
        # switch out with in on odd timesteps
        constants = self.sim.constants
        alloc_in = self.alloc_in
        alloc_out = self.alloc_out
        constants[alloc_in.name] = alloc_out.buffer
        constants[alloc_out.name] = alloc_in.buffer
    
    def pre_finalize(self):
        target = self.parent.spike_target
        target.reader = "(%s) + (%s*%s)" % (target.reader, str(self.weight), 
                                            self.name)
        
    def in_read_incoming_spikes(self, g):
        """
        name = reader
        reader = 0
        """ << g
