#!/bin/bash
python setup_env_util.py "$@"
chmod +x setup_env_util.sh
source setup_env_util.sh
rm setup_env_util.sh