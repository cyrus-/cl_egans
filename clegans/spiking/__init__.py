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
"""Spiking neural network simulations."""
import ahh.py as py
import ahh.cl as cl
from ahh.cl.egans import Node, StandaloneCode, Allocation

class State(Node):
    """A state variable in a spiking model node."""
    @py.autoinit
    def __init__(self, parent, basename, #@UnusedVariable
                 cl_dtype=cl.cl_float, #@UnusedVariable
                 calculations=None, calculations_hook="in_state_calculations", #@UnusedVariable
                 spike_updater=None, no_spike_updater=None, #@UnusedVariable
                 initializer=None): #@UnusedVariable
        pass
        
    def pre_finalize(self):
        self.code_node = StandaloneCode(self, hook=self.calculations_hook, 
                                        code=self.calculations)
                 
    cl_dtype = None
    """The data type of this state variable."""
                 
    calculations = None
    """If auxiliary calculations are needed, this string is placed in 
    ``calculations_hook``."""

    calculations_hook = None
    """The hook to place ``calculations`` into. Defaults to 
    ``in_state_calculations``."""

    spike_updater = None
    """The expression to use to update the state if there is a spike.
    
    If None, no update will be done.
    """

    no_spike_updater = None
    """The expression to use to update the state if there is not a spike.
    
    If None, no update will be done.
    """
    
    @property
    def using_independent_update(self):
        """Returns whether the update will be placed in the independent_update
        hook.
        
        This is the case if the spike and no_spike updaters are the same.
        """
        return self.spike_updater == self.no_spike_updater


    reader = "allocation[idx_state]"
    # The string to use to read the state variable.

    @py.lazy(property)
    def allocation(self):
        """After allocation, this will be the Allocation containing the state."""
        count = self.model.count * self.sim.n_realizations_per_division_max
        return Allocation(self, "buffer", (count,), self.cl_dtype)

    def in_read_state(self, g):
        """
        name = reader
        """ << g

    def in_independent_state_updates(self, g):
        if self.using_independent_update:
            independent_updater = self.spike_updater
            if independent_updater is not None:
                """
                allocation[idx_state] = spike_updater
                """ << g

    def in_spike_state_updates(self, g):
        if not self.using_independent_update:
            spike_updater = self.spike_updater
            if spike_updater is not None:
                """
                allocation[idx_state] = spike_updater
                """ << g
                
    def in_no_spike_state_updates(self, g):
        if not self.using_independent_update:
            no_spike_updater = self.no_spike_updater
            if no_spike_updater is not None:
                """
                allocation[idx_state] = no_spike_updater
                """ << g
    
    @property
    def _CG_expression(self):
        return self.name
    
    def on_initialize_memory(self, timestep_info): #@UnusedVariable
        initializer = self.initializer
        if initializer is not None:
            initializer(self.allocation.buffer, 
                        timestep_info.n_realizations*self.model.count)
            
class InitializeFromHost(Node):
    """Add to a :class:`State` node to specify that it be initialized by 
    copying a buffer initialized on the host. The ``array_producer`` attribute
    should be a function taking a shape and a numpy dtype and returning a 
    numpy array of that shape and dtype initialized as desired.
    """ 
    @py.autoinit
    def __init__(self, parent, array_producer, basename="InitializeFromHost"): 
        pass
    
    def on_finalize(self):
        def initializer(buffer, count):
            self.sim.ctx.memcpy(buffer, 
                self.array_producer((count,), buffer.infer_dtype(buffer)))
        self.parent.initializer = initializer
