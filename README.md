# mm_scripting_util

Utility class for simulation, training, and evaluation of parton-level processes. Provides a wrapper for madminer and madgraph functions. <br>
Provides a python object in `mm_scripting_util.core.miner` with  
  - awesome command line interface
  - quick addition/substitution of arbitrarily complex processes
  - setup and runtime wrappers for madminer and madgraph on various (frequently-used) servers
  - storage of multiple samples, augmented datasets, training models, and evaluation datasets within a single directory

TODO:
  - provide examples of madminer processes wrapped with this class
  - provide documentation
  
### General Usage
```
import mm_scripting_util as mm

tth_miner = mm.core.miner(
    name="unnamed",
    backend="tth.dat"
  )
  
tth_miner.simulate(
    samples=100000,
    benchmark='w',
    augmented_samples=10000,
    augmentation_benchmark='w'
  )

## etc, many more functions to add 

```
