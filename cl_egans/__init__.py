# TODO: Licensing

"""Provides infrastructure for generating and running Ace.OpenCL-accelerated 
simulations.

What is a Simulation?
*********************
A simulation in this context is a sequence of *elements*. The *state* of each 
element is updated every *timestep* according to a corresponding *model*.

For example, in a neurobiological circuit simulation, the elements are neurons,
and the state variables of the neurons (voltage, conductance, etc.) are updated
every timestep according to some formal neuron model (e.g. LIF, HH, etc.). 

Simulation Trees
****************
A family of simulations may have a common structure, but each particular 
simulation may require a slightly different set of capabilities. For example, 
you may want to probe different quantities on different runs. We need a way to 
build particular simulations using composable components.

In cl_egans, these composable components are called *nodes* and they are 
arranged in a tree. A node is an instance of the :class:`Node` class.

The root node of all simulations is an instance of the :class:`Simulation` 
class. Beneath the root node live the various :class:`Model` nodes as well as 
any simulation-wide utility nodes (to enable such things as random number 
generation for all nodes, for example.)

The :class:`Node` initializer takes as its first argument a parent node, and
adds itself to the tree upon initialization automatically. All nodes also have
a unique name, generated by providing a "basename" on initialization. See 
:class:`cypy.Naming` for more details.

Models
******
Model nodes control a contiguous range of elements, specified by their 
``count`` attribute.

For a spiking neuron simulation, for example, one model is the leaky
integrate-and-fire (LIF) model. If your simulation consists of 4000 LIF 
neurons, you would create one instance of your :class:`LIF` class and set its 
``count`` to 4000. It is given responsibility for elements with index 0 to 3999.

Each :class:`Model` node can in turn contain further children which
control aspects of that particular model (for example, spike propagation,
synapses, inputs, probes and so on). This motif repeats itself recursively
if these child nodes can be further configured.

A simulation can contain multiple models.

Hooks
*****
Each node is capable of triggering a *hook*. This is a function call 
that traverses the subtree from which it was invoked using a 
`depth-first traversal <https://secure.wikimedia.org/wikipedia/en/wiki/Depth-first_traversal>`_.

For example, the Simulation node triggers hooks corresponding to memory 
allocation, code generation, during runtime and for memory deallocation when 
asked by the script to do so. Child nodes are responsible for doing what they 
need to do during these hooks.

To trigger a hook from a Node, use ``self.trigger_hook(hookname, *args)``. 
Often, it is useful to have a *staged hook*, which means that you call three
hooks in a row of the form ``pre_hookname``, ``on_hookname`` and 
``post_hookname``. This is facilitated by the 
``trigger_staged_hook(hookname, *args)`` convenience method.

To respond to a hook, a node class can simply define a method with the 
appropriate name and arguments and when the corresponding hook is triggered, 
that method is called. If it is not defined, it is assumed that the node does 
not wish to do anything during that hook -- no error is raised.

Code Generation Hooks
~~~~~~~~~~~~~~~~~~~~~
A common type of hook is a code generation hook. Listeners receive an instance
of :class:`CG <cypy.cg.CG>` as an argument and are responsible for generating 
cl.oquence code to insert into the step function at the corresponding point. 
There are a large number of code generation hooks, but it is expected that most 
nodes will only implement one or a few of these. 

These hooks can be triggered using the ``trigger_cg_hook`` and 
``trigger_staged_cg_hook`` methods. Staged code generation hooks use the name 
``in_<hookname>`` instead of ``on_hookname``. A comment is inserted in the 
source of the form ``# ___<hookname>___`` unless the ``_print_hooks`` attribute 
is set to ``False``.

The code generator is set up to replace identifiers in generated code with the
value of the corresponding attribute, searched recursively up the tree to the
root. So, for example, the identifier "DT" will be replaced by the value of 
the attribute "DT" in Simulation, unless one of the downstream nodes have over-
written it.

Realizations
************
You can automatically run multiple realizations of the same tree in parallel. 
All realizations share the same memory *structure*, generated code and 
constants. However, they each have their own independent memory *contents* and 
can diverge from one another on the basis of the ``realization_num`` variable 
which is available to generated code.

The :attr:`Simulation.n_realizations` attribute controls the number of 
realizations.

.. Note:: Realizations cannot communicate with one another. Communication
          should only happen between elements of a single realization.
          
Divisions
*********
If the memory needed by the realizations exceeds the amount of storage
available on your device, the realizations can be split into divisions.

The realizations comprising a division are loaded onto the device, run to
completion and then the next division is loaded.

The :attr:`Simulation.n_realizations_per_division_max` attribute controls the
number of realizations per division. The final division may contain fewer than 
this many realizations if the number of realizations is not divisible by this
quantity. 

cl_egans?
*********
My primary purpose in designing this module was to accelerate spiking 
neurobiological circuit simulations (see :mod:`pyocl_egans.spiking`). 
`Caenorhabditis elegans`, or 
`c. elegans <http://en.wikipedia.org/wiki/Caenorhabditis_elegans>`_, is a model 
organism in neuroscience. It is one of the simplest organisms with a nervous 
system, and its entire 302-neuron network has been mapped out.

Examples
********
Examples are available in the ``test`` directory. Start with coba-brette07, 
which implements a standard benchmark (reference Brian implementation is also
provided, other implementations are available on ModelDB, see the paper.)

API
***

"""
import numpy
import cypy as py
import cypy.cg as cg
import clq.backends.opencl.pyocl as cl 

class Error(Exception):
    """Base class for errors in pyocl_egans."""
    pass

class Node(cg.Node):
    """A cl_egans Node. See the `base class <cypy.cg.Node>`_ for more 
    attributes.""" 
    @property
    def sim(self):
        """The :class:`Simulation` associated with this Node."""
        return self.getrec('sim', False)
    
    @property
    def model(self):
        """The :class:`Model` associated with this Node. 
        
        Raises an AttributeError if there is no :class:`Model` on the path to
        the root from this node.
        """
        return self.getrec('model', False)

class Simulation(Node):
    """The root node of a Simulation tree.
    
    **Steps**
    
    1. Finalization of the specification (:meth:`finalize`)
    
    2. Memory allocation (:meth:`allocate`)
    
    3. Code generation (:meth:`generate`)
    
    4. Runtime (:meth:`run`)
    
    5. Memory deallocation (:meth:`release`)
    
    """
    @py.autoinit
    def __init__(self, ctx, #@UnusedVariable
                 
                 n_realizations=1, #@UnusedVariable
                 n_realizations_per_division_max=1, #@UnusedVariable
                 DT=0.1, #@UnusedVariable,
                 n_timesteps=10000,  #@UnusedVariable
                 
                 # Sets it up as root:
                 parent=None, #@UnusedVariable
                 basename=None, #@UnusedVariable
                 **kwargs): #@UnusedVariable
        self.constants = { }
                 
    @property
    def sim(self):
        # Base case for getrec('sim')
        return self

    @py.setonce(property)
    def ctx(self):
        """The :class:`pyocl.Context` to bind this simulation to.
        
        If not provided during initialization, defaults to the process-wide
        context, :data:`pyocl.ctx`.
        """
        return self._ctx
    
    @ctx.setter
    def ctx(self, value): #@DuplicatedSignature
        if value is py.Default:
            value = cl.ctx
        self._ctx = value
        
    @property
    def Naming_prefix(self):
        # so Simulation_ is not added to all variables.
        return ""   
    
    DT = None
    """DT for Euler integration."""
    
    t = "DT*timestep"
    """Expression for calculating t based on DT and the current timestep."""
    
    n_timesteps = None
    """The total number of timesteps the simulation will run."""

    ## element counts relative to various things
    n_realizations = 1
    """The number of realizations of the simulation to run."""

    n_realizations_per_division_max = 1
    """The maximum number of realizations per division."""
    
    ############################################################################
    # Specification
    ############################################################################
    @property
    def models(self):
        """A tuple containing the models added so far.

        Models are a special subset of the nodes. They must:

        - Specify a ``count``, which means they are responsible for that many
          elms per realization of the network. Models are assigned indices
          sequentially in the order they are added.
        - They must have their own code generation function defined, which the
          Simulation hands control off to for the appropriate index range.

        See the :class:`Model` base class.
        """
        return tuple(node for node in self.children if isinstance(node, Model))

    @property
    def model_offsets(self):
        """A tuple containing the model index offsets."""
        return tuple(self._model_offsets())
    
    def _model_offsets(self):
        offset = 0
        for model in self.models:
            yield offset
            offset += model.count
            
    @property
    def model_offsets_map(self):
        """A map from models to their offsets."""
        return dict(zip(self.models, self._model_offsets()))

    @property
    def n_elms_per_realization(self):
        """Returns the total number of elements per realization.

        (i.e. the sum of model counts.)
        """
        return sum(model.count for model in self.models)

    @property
    def count(self):
        """An alternative name for :data:`n_elms_per_realization`."""
        return self.n_elms_per_realization

    @property
    def n_elms_per_sim(self):
        """Returns the number of elements the entire simulation is managing.

        (i.e. ``n_elms_per_realization * n_realizations``)
        """
        return self.n_elms_per_realization * self.n_realizations

    @property
    def n_elms_per_division_max(self):
        """Returns the maximum number of elements per division.

        (i.e. ``n_elms_per_realization * n_realizations_per_division_max``)
        """
        return self.n_elms_per_realization * \
               self.n_realizations_per_division_max

    @property
    def n_divisions(self):
        """Returns the number of divisions.

        See source for calculation.
        """
        return py.int_div_round_up(self.n_realizations,
                                   self.n_realizations_per_division_max)
        
    @property
    def idx_ranges(self):
        """Returns a map from the name of an index available in generated code
        to a tuple representing the range of values it can be.
        
        For example, idx_ranges['idx_sim'] ranges from 0 to n_elms_per_sim.
        """
        return {
            "idx_sim": (0, self.n_elms_per_sim),
            "idx_realization": (0, self.n_elms_per_realization),
            "idx_division": (0, self.n_elms_per_realization),
        }

    ############################################################################
    # Finalization
    ############################################################################
    def finalize(self):
        """Triggers finalization steps.

        Until this method has been called, all parameters for simulation units 
        are modifiable, without side effects, unless explicitly prohibited by 
        the unit.
        
        Calls the statged "finalize" hook the first time it is called. Does nothing 
        every subsequent time.
        """
        if not self.finalized:
            self.trigger_staged_hook('finalize')
            self._finalized = True
            
    @staticmethod
    def _assert(guard):
        if not guard:
            raise Error("Assertion failed, cannot continue.")

    def post_finalize(self):
        # Assertions that should hold for all specifications.
        self._assert(self.n_realizations > 0)
        self._assert(self.n_realizations_per_division_max > 0)
        self._assert(self.n_realizations_per_division_max <= self.n_realizations)
        
    @property
    def finalized(self):
        """Returns whether :meth:`finalize` has been called yet."""
        return getattr(self, '_finalized', False)

    ############################################################################
    # Memory Allocation
    ############################################################################
    def allocate(self):
        """Allocates (but does not initialize) memory for the simulation.

        If not already finalized, calls finalize first.
        
        Triggers the staged "allocate" hook the first time it is called. Does 
        nothing every subsequent time.
        """
        if not self.finalized:
            self.finalize()

        if not self.allocated:
            self.trigger_staged_hook("allocate")
            self._allocated = True

    @property
    def allocated(self):
        """Returns whether :meth:`allocate` has been called yet."""
        return getattr(self, '_allocated', False)
    
    def print_memory_summary(self):
        """Prints a plain text summary of the memory allocated by this 
        simulation."""
        ctx = self.ctx
        print "=== Memory summary for simulation running on", ctx.device.name, "==="
        self.trigger_hook("on_print_memory_summary")
        accumulator = py.Accumulator(0)
        self.trigger_hook("on_calculate_total_memory_usage", accumulator)
        print "%40s: %7.2f MB" % ("TOTAL", accumulator.value / 1024.0 / 1024.0)
    
    ############################################################################
    # Memory Release
    ############################################################################
    def release(self):    
        """Releases memory consumed by this simulation.
        
        If :meth:`allocate` has not been called, raises an :class:`Error`.
        Otherwise, triggers the staged "release" hook.
        
        After release, you can call :meth:`allocate` again.
        """
        if self.allocated:
            self.trigger_staged_hook('release')
            self._allocated = False
        else:
            raise Error("Memory was not allocated.")
        
    def on_release(self):
        # Goes through the list of constants and releases all buffer constants.        
        for constant in self.constants:
            if isinstance(constant, cl.Buffer):
                constant.release()
            self.ctx.release_all()

    ############################################################################
    # Code Generation
    ############################################################################
    def generate(self):
        """This step generates the cl.oquence code that will run on the device.

        If not finalized, calls finalize. Does NOT call allocate, even if it
        hasn't been called. If called multiple times, will generate code 
        multiple times.
        
        Triggers the "step_kernel" generation hook.
        
        After generation, the ``code`` attribute contains the cl.oquence code 
        produced. This is also returned.
        """
        g = self._make_code_generator()
        items = self.trigger_staged_cg_hook("step_kernel", g)
        g.append(items)
        code = self.code = g.code
        self._generated = True
        return code

    @py.lazy(property)
    def _step_fn_even(self):
        if not self.generated:
            self.generate()
        self.trigger_staged_hook("prepare_step_fn_even")        
        return cloquence.from_source(self.code, 
            constants=self.constants, size_calculator=self._size_calculator)
        
    @py.lazy(property)
    def _step_fn_odd(self):
        if not self.generated:
            self.generate()
        self.trigger_staged_hook("prepare_step_fn_odd")
        return cloquence.from_source(self.code, 
            constants=self.constants, size_calculator=self._size_calculator)
    
    @property
    def n_work_items(self):
        return min(int(py.ceil_int(self.n_elms_per_sim / 256.0)*256), 
                            self.ctx.device.max_work_items)
        
    @property
    def n_work_items_per_work_group(self):
        return 256
    
    @property
    def _size_calculator(self):
        # round to nearest multiple of 256        
        size_calculation = ((self.n_work_items,), 
                            (self.n_work_items_per_work_group,))
        return lambda timestep, realization_start: size_calculation #@UnusedVariable
        
    @property
    def generated(self):
        """Returns whether :meth:`generate` has been called yet."""
        return getattr(self, "_generated", False)
        
    def in_step_kernel(self, g):
        ("def step_fn(timestep, realization_start):\n", g.tab) >> g
        self.trigger_staged_cg_hook("step_kernel_body", g)
        
    def in_step_kernel_body(self, g):
        self.trigger_staged_cg_hook("thread_idx_calculations", g)
        self.trigger_staged_cg_hook("main_loop", g)
        
    def in_thread_idx_calculations(self, g):
        """
        gid = get_global_id(0)
        gsize = get_global_size(0)
        """ << g
        
    def in_main_loop(self, g):
        """
        first_idx_sim = realization_start * n_elms_per_realization
        last_idx_sim = min(first_idx_sim + n_elms_per_division_max, n_elms_per_sim)
        for idx_sim in (first_idx_sim + gid, last_idx_sim, gsize):
        """ << g
        g.append(g.tab)
        self.trigger_staged_cg_hook('loop_body', g)
        
    def in_loop_body(self, g):
        self.trigger_staged_cg_hook("element_idx_calculations", g)
        p = cg.Partitioner(g.append, "idx_realization",
                           min_start=0, max_end=self.n_elms_per_realization)
        for model in self.models:
            p.next(start=model.offset, end=model.offset + model.count,
                   code=lambda g: model.generate_step_kernel(g))
                
    def in_element_idx_calculations(self, g):
        """
        realization_num = idx_sim / n_elms_per_realization
        realization_first_idx_sim = realization_num * n_elms_per_realization
        realization_first_idx_div = (realization_num - realization_start)*n_elms_per_realization
        idx_realization = idx_sim - realization_first_idx_sim
        idx_division = idx_sim - first_idx_sim
        """ << g

    ############################################################################
    # Runtime
    ############################################################################
    def run(self):
        """Runs the simulation for the specified number of timesteps.
        
        1. The "prepare_run" hook is triggered with an instance of 
           :class:`RunInfo`.
           
        2. When a division is loaded, the first step is to ask each node to 
           initialize its allocations via the "on_initialize_memory" hook. They
           are passed an instance of :class:`TimestepInfo`.
           
        3. After each timestep is complete, the "on_timestep_complete" hook is
           triggered with the :class:`TimestepInfo` instance.
           
        4. After each division is complete, the "on_division_complete" hook is
           triggered with the :class:`TimestepInfo` instance.
           
        5. After the simulation is complete and the Context's queue is flushed,
           the "on_run_complete" hook is triggered with the :class:`RunInfo`
           instance.
        """
        n_timesteps = self.n_timesteps
        step_fn_even = self._step_fn_even
        step_fn_odd = self._step_fn_odd
        
        run_info = self.RunInfo(n_timesteps)
        
        # 1.
        self.trigger_hook("prepare_run", run_info)
        
        for division_num in xrange(self.n_divisions):
            max_realizations = self.n_realizations_per_division_max
            realization_start = numpy.int32(division_num * max_realizations)
            n_realizations = realization_start + max_realizations
            total_realizations = self.n_realizations
            if n_realizations > total_realizations:
                n_realizations = total_realizations - realization_start
            timestep_info = self.TimestepInfo(run_info, division_num,
                realization_start, n_realizations)
            
            # 2.
            self.trigger_hook("on_initialize_memory", timestep_info)
            
            timestep = numpy.int32(0)
            while timestep < n_timesteps:                
                timestep_info.timestep = timestep
                
                if timestep % 2 == 0:
                    step_fn_even(timestep, realization_start)
                else:
                    step_fn_odd(timestep, realization_start)
                
                self.trigger_hook("on_timestep_complete", timestep_info)
                
                timestep += 1
            self.trigger_hook("on_division_complete", timestep_info)
            
        self.ctx.queue.finish() # wait for everything to complete
        self.trigger_hook("on_run_complete", run_info)
        
    class RunInfo(object):
        """Contains information associated with a call to :meth:`run`."""
        def __init__(self, n_timesteps):
            self.n_timesteps = n_timesteps
            
        n_timesteps = None
        """The number of timesteps."""
            
    class TimestepInfo(object):
        """Contains information associated with a single timestep for a single 
        division."""
        @py.autoinit
        def __init__(self, run_info, division_num, realization_start, 
                     n_realizations):
            pass
        
        timestep = 0
        
    ############################################################################
    # RNG
    ############################################################################
    @py.lazy(property)
    def rng(self):
        """Lazily produces a :class:`RNG` node available for use by all models."""
        return RNG(self)
    
    @property
    def randf(self):
        self.rng # initialize RNG if not already
        return "randf"
    
    @property
    def randexp(self):
        self.rng # initialize RNG if not already
        return "randexp"
    
    @property
    def randn(self):
        self.rng # initialize RNG if not already
        return "randn"

class Model(Node):
    """Specifies the behavior of a contiguous range of elements."""
    
    @py.autoinit
    def __init__(self, parent, count=1, basename="Model"): pass
    
    @property
    def model(self):
        # Base case for getrec('model')
        return self
    
    count = None
    """The number of elements to control using this Model."""
    
    @property
    def offset(self):
        """The index of the first element in the simulation controlled using 
        this Model."""
        return self.sim.model_offsets_map[self]
    
    @property
    def idx_ranges(self):
        """Augments :data:`Simulation.idx_ranges` with additional indices
        available to this model."""
        idx_ranges = dict(self.sim.idx_ranges)
        idx_ranges['idx_model'] = (0, self.count)
        return idx_ranges
    
    ## Model Code Generation
    def generate_step_kernel(self, g):
        self.trigger_staged_cg_hook("model_idx_calculations", g)
        self.trigger_staged_cg_hook("model_cl_code", g)
        
    def in_model_idx_calculations(self, g):
        """
        idx_model = idx_realization - offset
        """ << g

class MemoryNode(Node):
    """Represents a Node containing a memory element.
    
    Subclasses of MemoryNode, like :class:`Allocation`, :class:`ConstantArray`
    and :class:`Array`, override the :data:`fn` method, which specifies which
    of the memory management functions in :class:`pyocl.Context` will be used.
    
    See source of the classes for details.
    """
    def __init__(self, parent, basename, *args, **kwargs):
        cg.Node.__init__(self, parent=parent, basename=basename, args=args,
                      kwargs=kwargs)
        
    @property
    def _CG_expression(self):
        # for proper substitution
        return self.name
    
    @py.lazy(property)
    def buffer(self):
        """The :class:`pyocl.Buffer` corresponding to this memory node."""
        buffer = self.fn(*self.args, **self.kwargs)
        self.sim.constants[self.name] = buffer
        return buffer
       
    def on_allocate(self):
        self.buffer # make sure its been created
        
    def on_print_memory_summary(self):
        buffer = self.buffer
        print "%40s: %7.2f MB [%s x %s]" % (self.name, 
            buffer.size / 1024.0 / 1024.0, 
            buffer.shape, buffer.cl_dtype.name)
        
    def on_calculate_total_memory_usage(self, accumulator):
        accumulator += self.buffer.size
    
class Allocation(MemoryNode):
    """Represents an uninitialized memory allocation, using Context.alloc."""
    @property
    def fn(self):
        """:meth:`pyocl.Context.alloc`"""
        return self.sim.ctx.alloc
    
class ConstantArray(MemoryNode):
    """Represents a constant array over all realizations, using Context.In."""
    @property
    def fn(self):
        """:meth:`pyocl.Context.In`"""
        return self.sim.ctx.In
    
class Array(MemoryNode):
    """Represents a variable array over all realizations, using 
    Context.to_device.
    """
    @property
    def fn(self):
        """:meth:`pyocl.Context.to_device`"""
        return self.sim.ctx.to_device
    
#class RNG(Node):
#    """Enables random number generation. See :data:`Simulation.rng`.
#    
#    Once enabled, the functions ``randf``, ``randexp`` and ``randn`` can be
#    used in generated code.
#    """
#    
#    @py.autoinit
#    def __init__(self, parent, basename="RNG"): pass
#    
#    randf = clements.simple_randf
#    """The base random number function to use. Defaults to 
#    :data:`CLements.simple_randf`."""
#    
#    initializer = None
#    # The initializer to use for the random number generator's state. Defaults
#    # to randf.initializer.
#    
#    state = None
#    """Contains the state allocation after finalization."""
#    
#    def post_allocate(self):
#        sim = self.sim
#        self.state = Allocation(self, "state", 
#            (sim.n_work_items,), cl.cl_int)
#        
#        if self.initializer is None:
#            self.initializer = self.randf.initializer
#
#        randf = self.randf
#        state = self.state.buffer
#        randf_bound = self.randf_bound = cloquence.fn(randf, 
#            state=state)
#        
#        sim.constants['randf'] = randf_bound
#        sim.constants['randexp'] = cloquence.fn(clements.randexp, 
#                                                randf=randf, state=state)
#        sim.constants['randn'] = cloquence.fn(clements.randn, 
#                                              randf=randf, state=state)
#        
#    def on_initialize_memory(self, timestep_info): #@UnusedVariable
#        sim = self.sim
#        sim.ctx.memcpy(self.state.buffer, self.initializer(sim.n_work_items))
        
class Probe(Node):
    """Abstract base class for all data probes."""
    
    @py.autoinit
    def __init__(self, parent, basename="Probe"): pass

class ConstrainedProbe(Probe):
    """Abstract base class for probes which are only activated for a 
    constrained range of elements and timesteps."""
    
    @py.autoinit
    def __init__(self, parent, basename="ConstrainedProbe",
                 t_range=(0, None, 1),
                 idx="idx_model",
                 idx_range=(0, None, 1)): pass
                 
    def on_finalize(self):
        t_range = list(self.t_range)
        if t_range[1] is None:
            t_range[1] = self.sim.n_timesteps
        if len(t_range) == 2:
            t_range.append(1)
        self.t_range = tuple(t_range)
        
        idx_range = list(self.idx_range)
        if idx_range[1] is None:
            max_idx = self.getrec("idx_ranges")[self.idx][1]
            idx_range[1] = max_idx
        if len(idx_range) == 2:
            idx_range.append(1)
        self.idx_range = tuple(idx_range)
        
    t_range = None
    """The range (start, stop, step) of timesteps."""
    
    idx_range = None
    """The range (start, stop, step) of indices."""
    
    idx = None
    """The index to use for constraining. Defaults to "idx_model"."""

    @property
    def t_start(self):
        """t_range[0]"""
        return self.t_range[0]
    
    @property
    def t_stop(self):
        """t_range[1]"""
        return self.t_range[1]
    
    @property
    def t_step(self):
        """t_range[2]"""
        return self.t_range[2]
    
    @property
    def idx_start(self):
        """idx_range[0]"""
        return self.idx_range[0]
    
    @property
    def idx_stop(self):
        """idx_range[1]"""
        return self.idx_range[1]
    
    @property
    def idx_step(self):
        """idx_range[2]"""
        return self.idx_range[2]

    @property
    def total_n_timesteps(self):
        """int((t_stop - t_start)/t_step)"""
        t_range = self.t_range
        return int((t_range[1] - t_range[0])/t_range[2])
    
    @property
    def n_elms(self):
        """int((idx_stop - idx_start)/idx_step)"""
        idx_range = self.idx_range
        return int((idx_range[1] - idx_range[0]/idx_range[2]))
    
    @property
    def n_realizations(self):
        return self.sim.n_realizations_per_division_max
    
    def constrain(self, g):
        """Produces the if statement corresponding to the provided constraints
        on the provided code generator."""
        self.constraints = constraints = tuple(self._yield_constraints())
        if constraints:
            constraints = " and ".join(constraints)
            ("if %s:\n" % constraints, g.tab) >> g
            
    def unconstrain(self, g):
        """Closes the if statement."""
        constraints = self.constraints
        if constraints:
            g.untab >> g
            
    def _yield_constraints(self):
        for constraint in self._yield_t_constraints():
            yield constraint
            
        for constraint in self._yield_idx_constraints():
            yield constraint
            
    def _yield_t_constraints(self):
        if self.t_start > 0:
            yield "timestep >= t_start"
        
        if self.t_stop != self.sim.n_timesteps:
            yield "timestep < t_stop"
        
        if self.t_step != 1:
            yield "timestep % t_step == 0"
        
    def _yield_idx_constraints(self):
        if self.idx_start > 0:
            yield "idx >= idx_start"
            
        if self.idx_stop != self.getrec("idx_ranges")[self.idx][1]:
            yield "idx < idx_stop"
            
        if self.idx_step != 1:
            yield "idx % idx_step == 0"
            
class PerElementProbe(ConstrainedProbe):
    """Abstract base class for probes which store data in a 3D matrix, 
    indexed by (timestep, realization number, element index).
    
    Can specify how many timepoints to actually buffer and a hook is called
    when the buffer is full ("on_buffer_full").    
    """
    
    @py.autoinit
    def __init__(self, parent, basename="PerElementProbe",
                 buffer_timepoints=None,
                 cl_dtype=cl.cl_float): 
        pass
    
    buffer_timepoints = None
    """The number of timepoints to store before triggering the on_buffer_full 
    hook."""
    
    @property
    def shape(self):
        """The shape of the buffer."""
        n_realizations = self.sim.n_realizations_per_division_max
        buffer_timepoints = self.buffer_timepoints
        n_elms = self.n_elms
        return buffer_timepoints, n_realizations, n_elms

    @py.lazy(property)
    def allocation(self):
        """The :class:`Allocation` containing the buffer."""
        return Allocation(self, "buffer", self.shape, self.cl_dtype)
    
    def on_finalize(self):
        super(PerElementProbe, self).on_finalize()
        
        self._infer_buffer_timepoints()
        self.allocation
        
    def _infer_buffer_timepoints(self):
        if self.buffer_timepoints is None:
            self.buffer_timepoints = self.total_n_timesteps
            
        buffer_timepoints = self.buffer_timepoints
        total_n_timesteps = self.total_n_timesteps
        if buffer_timepoints != total_n_timesteps:
            assert total_n_timesteps % buffer_timepoints == 0
            self.timestep_expr = "%s %% buffer_timepoints" % self.timestep_expr
            
    def on_allocate(self):
        self.allocation # make sure its been accessed

    def on_timestep_complete(self, timestep_info):
        timestep = timestep_info.timestep
        if timestep < self.t_stop:
            timesteps_elapsed = self.timesteps_elapsed(timestep_info.timestep)
            if (timesteps_elapsed > 0 and   
                timesteps_elapsed % self.buffer_timepoints == 0):
                self.trigger_hook("on_buffer_full", timestep_info, 
                                  timesteps_elapsed)
                
    def timesteps_elapsed(self, timestep):
        return (timestep - self.t_start)/self.t_step + 1
    
    buffer_idx_expression = "(timestep_expr*n_realizations + realization_expr)*n_elms + idx_expr"
    realization_expr = "realization_num - realization_start"
    timestep_expr = "(timestep - t_start)/t_step"
    idx_expr = "(idx - idx_start)/idx_step"

class ProcessOnHost(Node):
    """Responds to the on_buffer_full message by copying the buffer to the host
    for processing. 
    
    Should be added to a :class:`PerElementProbe`. A 
    :class:`Listener <cypy.cg.Listener>` can be used to specify how to 
    process the data after it has been copied by listening downstream on the
    same hook.
    """
    
    @py.autoinit
    def __init__(self, parent, basename="ProcessOnHost"): pass
    
    def post_allocate(self):
        parent = self.parent
        shape = parent.shape
        dtype = cl.Buffer.infer_dtype(parent.allocation.buffer)
        self.data_buffer = numpy.empty(shape, dtype)
        
    def on_buffer_full(self, run_info, timesteps_elapsed): #@UnusedVariable
        parent = self.parent
        ctx = self.sim.ctx
        data_buffer = self.data_buffer
        ctx.memcpy(data_buffer, parent.allocation.buffer)
        self.data = data_buffer
        parent.trigger_hook("on_process_data", data_buffer, self)

class AccumulateOnHost(Node):
    """Responds to all on_buffer_full messages by copying the data to the host
    in a host buffer sized for the whole simulation, which can be accessed
    after the simulation for processing."""
    
    @py.autoinit
    def __init__(self, parent, basename="AccumulateOnHost"): pass
    
    def on_allocate(self):
        parent = self.parent
        total_n_timesteps = parent.total_n_timesteps
        n_realizations = self.sim.n_realizations
        n_elms = parent.n_elms
        shape = (total_n_timesteps, n_realizations, n_elms)
        dtype = cl.Buffer.infer_dtype(parent.allocation.buffer)
        self.data_buffer = numpy.empty(shape, dtype)
    
    def on_buffer_full(self, timestep_info, timesteps_elapsed): #@UnusedVariable
        parent = self.parent
        ctx = self.sim.ctx
        timestep_start = timesteps_elapsed - parent.buffer_timepoints
        division_start = timestep_info.realization_start
        division_end = timestep_info.n_realizations + division_start
        data_buffer = self.data_buffer
        ctx.memcpy(data_buffer[timestep_start:timesteps_elapsed, 
                               division_start:division_end, 
                               :], parent.allocation.buffer)
        self.data = data_buffer
        parent.trigger_hook("on_process_data", data_buffer, self)

class ExpressionProbe(PerElementProbe):
    """A :class:`PerElementProbe` which records the value of the provided expression
    for each element during the provided hook."""
    
    @py.autoinit
    def __init__(self, parent, basename="ExpressionProbe",
                 
                 expression=None,
                 hook=None): pass
        
    def on_finalize(self):
        super(ExpressionProbe, self).on_finalize()
        
        assert self.expression
        assert self.hook
        
        setattr(self, self.hook, self._insert_code)
        
    def _insert_code(self, g):
        self.constrain(g)
        """
        allocation[buffer_idx_expression] = expression
        """ << g
        self.unconstrain(g)
        