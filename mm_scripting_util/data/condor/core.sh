#!/bin/bash

# CONDA_PATH = "/afs/cern.ch/work/l/llepotti/anaconda3"

# # __conda_setup="$('$CONDA_PATH/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"

# # eval "$__conda_setup"
# # else 
# #     if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
# #         . "$CONDA_PATH/etc/profile.d/conda.sh"
# #     else
# #         export PATH="$CONDA_PATH/etc/profild.d/conda.sh"
# #     fi
# # fi

# # unset __conda_setup

conda deactivate
conda activate atlas

python -m mm_scripting_util.run $@