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

import numpy
import cypy as py
from cl_egans import Node
from cl_egans.spiking import State, InitializeFromHost

class Current(Node):
    """Specifies a current."""
    
    @py.autoinit
    def __init__(self, parent, basename="Current"): pass
    
    current = None
    """Current magnitude."""
    
    def in_calculate_inputs(self, g):
        """
        input_current += current
        """ << g
    
class GenericSynapse(Current):
    """A generic synapse."""
    
    @py.autoinit
    def __init__(self, parent, basename="GenericSynapse", 
                current="g*(reversal - v)", reversal=None): pass
    
    current = "g*(reversal - v)"
    """Synaptic current."""
    
    reversal = None
    """Reversal potential."""
    
    spike_target = None
    """A reference to the state variable into which spikes should be sent."""
    
class LocalPoisson(Node):
    """Injects spikes via a homogeneous Poisson process into the parent synapse."""
    @py.autoinit
    def __init__(self, parent, basename="LocalPoisson", rate=1): pass
    
    rate = None
    """The rate, in Hz, of the Poisson process."""
    
    @property
    def rate_mHz(self):
        """Returns the rate in mHz."""
        return self.rate / 1000.0
    
    @property
    def reciprocal_rate_mHz(self):
        # Returns 1/rate_mHz
        return 1.0/self.rate_mHz
    
    weight = 1
    """The weight of a spike."""
    
    @py.lazy(property)
    def next_spike(self):
        state = State(self, "next_spike", spike_updater=None, 
                      no_spike_updater=None)
        initializer = InitializeFromHost(state, 
            array_producer=lambda count, dtype: numpy.random.exponential(
                self.rate_mHz, count).astype(dtype))
        return state
        
    @property
    def next_spike_alloc(self):
        return self.next_spike.allocation
    
    def on_finalize(self):
        self.next_spike
        
    def in_calculate_inputs(self, g):
        """
        if t >= next_spike:
            spike_target += weight
            isi = randexp()*reciprocal_rate_mHz
            while isi < DT: # high rate processes may produce >1 spike/timestep
                spike_target += weight
                isi += randexp()*reciprocal_rate_mHz
            next_spike_alloc[idx_state] = next_spike + isi
        """ << g
        
class ExponentialSynapse(GenericSynapse):
    """A synapse which produces exponential-shaped PSPs."""
    
    @py.autoinit
    def __init__(self, parent, basename="ExponentialSynapse",
                tau=None, reversal=None): pass
    
    @py.lazy(property)
    def g(self):
        """Conductance state variable."""
        return State(self, "g", spike_updater=None, no_spike_updater=None)
    
    @property
    def spike_target(self):
        return self.g
    
    def pre_finalize(self):
        g = self.g
        g.spike_updater = g.no_spike_updater = self.g_updater
    
    tau = None
    """Synaptic integration time constant."""
    
    g_updater = "g - DT/tau*g"
    """Synaptic conductance updater."""

