import os
import sys
import subprocess 

def _conda_info(
):
    """
    Returns the current os' installed conda environments, as well as the current env
    Returns empty list, None if conda is not installed.
    """
    python_version = sys.version_info[0:2]

    subprocess_object = subprocess.Popen(["conda env list"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    conda_ret_raw = [str(line.decode('utf8')) for line in (subprocess_object.stdout.readlines() + subprocess_object.stderr.readlines())]


    conda_ret = [line.strip("\n") for line in conda_ret_raw if line[0] != '#' and line != '\n']
    
    # handle uninstalled conda cases
    for line in conda_ret:
        if "conda: command not found" in line:
            return [], None

    # grab all/current envs
    conda_ret = [line.split() for line in conda_ret]
    conda_envs = [line[0] for line in conda_ret]
    conda_env_current = conda_ret["*" in conda_ret][0]
    return conda_envs, conda_env_current
