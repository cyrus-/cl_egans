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
"""Probes live here."""

import numpy
import ahh.py as py
import ahh.cl as cl
import ahh.cl.ements as clements
from ahh.cl.egans import ConstrainedProbe, PerElementProbe, Allocation

class SpikeRasterProbe(PerElementProbe):
    """A :class:`PerElementProbe <ahh.cl.egans.PerElementProbe>` which records
    a 0 for no spike and a 1 for a spike for each element."""
    
    @py.autoinit
    def __init__(self, parent, basename="SpikeRasterProbe",
                 cl_dtype=cl.cl_int): pass
    
    def pre_spike_generated(self, g):
        self.constrain(g)
        """
        allocation[buffer_idx_expression] = 1
        """ << g
        self.unconstrain(g)
        
    def pre_no_spike_generated(self, g):
        self.constrain(g)
        """
        allocation[buffer_idx_expression] = 0
        """ << g
        self.unconstrain(g)

class SpikeListProbe(PerElementProbe):
    """A sparse :class:`PerElementProbe <ahh.cl.egans.PerElementProbe>` which
    records the list of neurons which spiked at each timestep. 
    
    The count can be accessed via the ``counts`` attribute in the processor you 
    add (e.g. :class:`ProcessOnHost <ahh.cl.egans.ProcessOnHost>`.
    """
    @py.autoinit
    def __init__(self, parent, basename="SpikeListProbe",
                 cl_dtype=cl.cl_uint): pass

    def on_finalize(self):
        super(SpikeListProbe, self).on_finalize()
        self.count_allocation = Allocation(self, "count", 
           (self.buffer_timepoints, self.n_realizations), cl.cl_uint)
        
    def on_initialize_memory(self, timestep_info): #@UnusedVariable
        clements.ew_set_0(self.count_allocation.buffer)
        
    def pre_spike_generated(self, g):
        self.constrain(g)
        """
        n_spikes = atom_inc(count_allocation + (realization_num - realization_start)*buffer_timepoints + timestep_expr)
        allocation[buffer_idx_expression] = idx
        """ << g
        self.unconstrain(g)
        
    idx = "idx_model"
    idx_expr = "n_spikes"
    
    def on_process_data(self, data, mode):
        counts = self.sim.ctx.from_device(self.count_allocation.buffer)
        shape = data.shape
        new_data = numpy.ndarray((shape[0], shape[1]), dtype=object)
        for t, timeslice in enumerate(data):
            for r, realization in enumerate(timeslice):
                count = counts[t, r]
                new_data[t, r] = realization[0:count]
        
        mode.counts = counts        
        mode.data = new_data
        
class SpikeScatterProbe(ConstrainedProbe):
    """Produces a buffer containing spike times and another buffer containing
    spike indices which can be directly used to plot a raster plot."""
    
    @py.autoinit
    def __init__(self, parent, basename="SpikeScatterProbe"): pass
    
    max_spikes = None
    """The maximum number of spikes. Defaults to one per timestep possible
    (almost certainly an overestimate.)"""
    
    def on_finalize(self):
        super(SpikeScatterProbe, self).on_finalize()
        
        max_spikes = self.max_spikes
        if max_spikes is None:
            max_spikes = self.max_spikes = (self.total_n_timesteps *
                self.n_elms * self.n_realizations)
        self.count_allocation = Allocation(self, "count", (1,), cl.cl_uint)
        self.spike_times_allocation = Allocation(self, "spike_times", 
             (max_spikes,), self.time_expr_cl_dtype)
        self.spike_indices_allocation = Allocation(self, "spike_indices", 
             (max_spikes,), cl.cl_uint)
        
    def on_initialize_memory(self, timestep_info): #@UnusedVariable
        clements.ew_set_0(self.count_allocation.buffer)
        
    def pre_spike_generated(self, g):
        self.constrain(g)
        """
        n_spikes = atom_inc(count_allocation)
        spike_times_allocation[n_spikes] = time_expr
        spike_indices_allocation[n_spikes] = idx
        """ << g
        self.unconstrain(g)
        
    idx = "idx_model - idx_start"
    time_expr = "timestep - t_start"
    time_expr_cl_dtype = cl.cl_uint  # should be inferrable but not yet
    
    def get_data(self):
        """Return the spike_times and spike_indices."""
        get = self.sim.ctx.from_device
        count = get(self.count_allocation.buffer)[0]
        # With OpenCL 1.1 or CUDA we can get copy the first `count` elements
        spike_times = get(self.spike_times_allocation.buffer)[:count]
        spike_indices = get(self.spike_indices_allocation.buffer)[:count]
        return spike_times, spike_indices
    
    def plot(self, **kwargs):
        """Produce a raster plot. See :func:`ahh.np.plotting.raster`."""
        from ahh.np.plotting import raster
        times, indices = self.get_data()
        raster(times, indices, self.total_n_timesteps, self.n_elms, **kwargs)
            
class BinnedSpikeCountProbe(PerElementProbe):
    """A :class:`PerElementProbe` which produces spike counts in possibly
    overlapping bins instead of at every timestep."""
    
    @py.autoinit
    def __init__(self, parent, basename="BinnedSpikeCountProbe",
                 
                 bin_size=1,
                 shift_size=1,
                 
                 cl_dtype=cl.cl_uint,
                 ): pass
    
    def on_finalize(self):
        super(BinnedSpikeCountProbe, self).on_finalize()
        
        bin_size = self.bin_size
        shift_size = self.shift_size
        assert bin_size % shift_size == 0
        assert bin_size >= shift_size
        assert self.t_step == 1
        
    def on_initialize_memory(self, timestep_info): #@UnusedVariable
        clements.ew_set_0(self.allocation.buffer)
        
    @property
    def n_bins(self):
        return self.n_timesteps / self.shift_size
    
    @property
    def n_bins_per_spike(self):
        return self.bin_size / self.shift_size
    
    @property
    def total_n_timesteps(self):
        n_timesteps = self.t_stop - self.t_start
        shift_size = self.shift_size
        assert n_timesteps % shift_size == 0
        return n_timesteps / self.shift_size
    
    def timesteps_elapsed(self, timestep):
        return (timestep - self.t_start + 1)//self.shift_size
                 
    def pre_spike_generated(self, g):
        self.constrain(g)
        """
        bin = (timestep - t_start)/shift_size
        atom_inc(allocation + buffer_idx_expression)
        """ << g
        for _ in xrange(self.n_bins_per_spike - 1):
            """
            bin -= 1
            if bin >= 0:
                atom_inc(allocation + buffer_idx_expression)
            """ << g
            g << g.tab
            
        for _ in xrange(self.n_bins_per_spike - 1):
            g << g.untab
            
        self.unconstrain(g)
            
    timestep_expr = "bin"
    