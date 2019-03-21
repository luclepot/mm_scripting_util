# mm_scripting_util

Utility class for simulation, training, and evaluation of parton-level processes. Provides a wrapper for madminer and madgraph functions. <br>
Provides a python object in `mm_scripting_util.core.miner` with  
  - quick addition/substitution of arbitrarily complex processes
  - setup and runtime wrappers for madminer and madgraph on various (frequently-used) servers
  - storage of multiple samples, augmented datasets, training models, and evaluation datasets within a single directory
TODO:
  - implement/fix current command line interface for python object
  - provide examples of madminer processes wrapped with this class
  
