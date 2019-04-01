#!/bin/bash
python mm_scripting_util/setup_env_util.py "$@"
if [ -e setup_env_util.sh ]; then 
    echo "running setup script..."
    chmod +x setup_env_util.sh;
    source setup_env_util.sh;
    rm setup_env_util.sh;
fi
echo "done"
