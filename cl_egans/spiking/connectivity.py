"""Connectivity and communication stuff lives here."""

import cypy as py
import clq.backends.opencl as clqcl
import clq.stdlib as clqstd
from cl_egans import Node, Allocation

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
        g << 'exec "' << clqcl.cl_khr_global_int32_base_atomics.pragma_str << '"\n'
    
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
        return Allocation(self, "in", self._buffer_size, clqcl.int)
    
    @py.lazy(property)
    def buffer_in(self):
        return self.alloc_in.buffer
    
    @py.lazy(property)
    def alloc_out(self):
        return Allocation(self, "out", self._buffer_size, clqcl.int)
    
    @py.lazy(property)
    def buffer_out(self):
        return self.alloc_out.buffer
    
    def on_initialize_memory(self, timestep_info):
        clqstd.ew_set_0(self.buffer_in)
        clqstd.ew_set_0(self.buffer_out)
    
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
