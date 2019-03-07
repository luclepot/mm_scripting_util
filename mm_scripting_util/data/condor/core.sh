#!/bin/bash

# edit to change conda path, if needed
__conda_setup="$('/afs/cern.ch/work/l/llepotti/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"

# change conda setups
if [$? -eq 0]; then
    eval "$__conda_setup"
    else
        if [ -f "/afs/cern.ch/work/l/llepotti/anaconda3/etc/profile.d/conda.sh" ]; then
            . "/afs/cern.ch/work/l/llepotti/anaconda3/etc/profile.d/conda.sh"
        else
            export PATH="/afs/cern.ch/work/l/llepotti/anaconda3/etc/profild.d/conda.sh"
        fi
    fi
    unset __conda_setup

conda deactivate
conda activate atlas

python -m mm_scripting_util.run $@
