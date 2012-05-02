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
"""Neuron models live here."""
import numpy
import cypy as py
from cl_egans import Model
from cl_egans.spiking import State, InitializeFromHost

class SpikingModel(Model):
    """The base class of all spiking models."""
    @py.autoinit
    def __init__(self, parent, basename="SpikingModel", count=1): pass
    
    spike_condition = None
    """The expression to evaluate to determine whether a spike occurred."""
    
    def in_model_cl_code(self, g):
        """
        idx_state = idx_model + (realization_num - realization_start)*count
        """ << g
        self.trigger_staged_cg_hook("read_incoming_spikes", g)
        self.trigger_staged_cg_hook("read_state", g)
        self.trigger_staged_cg_hook("calculate_inputs", g)
        self.trigger_staged_cg_hook("state_calculations", g)
        self.trigger_staged_cg_hook("independent_state_updates", g)
        self.trigger_staged_cg_hook("spike_processing", g)
        
    def in_calculate_inputs(self, g):
        """
        input_current = 0
        """ << g
        
    def in_spike_processing(self, g):
        g << ("if spike_condition:\n", g.tab)
        self.trigger_staged_cg_hook("spike_generated", g)
        g << ("pass # in case no one writes out any code in this branch\n", g.untab)
        g << ("else:\n", g.tab)
        self.trigger_staged_cg_hook("no_spike_generated", g)
        g << ("pass # in case no one writes out any code in this branch\n", g.untab)
        
    def in_spike_generated(self, g):
        self.trigger_staged_cg_hook("spike_state_updates", g)
        self.trigger_staged_cg_hook("spike_propagation", g)
        
    def in_no_spike_generated(self, g):
        self.trigger_staged_cg_hook("no_spike_state_updates", g)
        
class GenericIF(SpikingModel):
    """A generic integrate-and-fire model with absolute refractory period. 
    
    Subclasses should specify the form of the leak using the ``leak`` attribute 
    to make a specific integrate-and-fire model.
    """
    
    @py.autoinit
    def __init__(self, parent, basename="GenericIF", count=1, 
                 v_update_eqn = "v + DT/tau*(leak + input_current)",
                 tau=None,
                 v_reset=None,
                 v_thresh=None,
                 abs_refractory_period=None): pass #@UnusedVariable
    
    @py.lazy(property)
    def v(self):
        """Voltage state variable."""
        
        return State(self, "v",
            calculations="v_new = v_update_eqn if not abs_refractory_condition else v_reset",
            spike_updater="v_reset",
            no_spike_updater="v_new")
        
    @py.lazy(property)
    def abs_refractory_t_release(self):
        """Absolute refractory period state variable."""
        
        state = State(self, "abs_refractory_t_release",
                     spike_updater="t + abs_refractory_period")
        InitializeFromHost(state,
            lambda shape, dtype: numpy.zeros(shape, dtype))
        return state
        
    def pre_finalize(self):
        self.v # lazy stuff needs to be created by this point
        self.abs_refractory_t_release
        # TODO: (post-init hook is really where this should go but its not implemented yet)
        
    v_update_eqn = None
    """Update equation for voltage."""
    
    tau = None
    """Membrane time constant."""
    
    v_reset = None
    """Post-spike reset potential."""
    
    v_thresh = None
    """Spike threshold potential."""
    
    spike_condition = "v_new >= v_thresh"
    """Condition for spiking."""
    
    abs_refractory_period = None
    """Absolute refractory period length."""
    
    abs_refractory_condition = "t < abs_refractory_t_release"
    """Boolean condition for absolute refractoriness."""
    
class ReducedLIF(GenericIF):
    """Reduced leaky integrate-and-fire with absolute refractory period.
    
    The leak is "-v".
    """
    
    @py.autoinit
    def __init__(self, parent, basename="LIF", count=1): pass

    leak = "-v"
