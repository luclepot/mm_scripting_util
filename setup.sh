#!/bin/bash
echo ''
if ! [ -x "$(command -v python)" ]; then
    echo 'setup.sh: no python on path; attempting install...'
    apt-get install python-pip
fi
python3 mm_scripting_util/setup_env_util.py "$@" || python mm_scripting_util/setup_env_util.py "$@"
if [ -e setup_env_util.sh ]; then 
    echo 'starting bash setup script run...'
    chmod +x setup_env_util.sh;
    source setup_env_util.sh;
    rm setup_env_util.sh;
    echo 'finished bash setup setup run.'
fi
echo ''