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

    SPobj = subprocess.Popen(["conda env list"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    conda_ret_raw = SPobj.stdout.readlines() + SPobj.stderr.readlines()


    
    conda_ret = [line.strip("\n").split() for line in conda_ret_raw if line[0] is not "#" and line is not "\n"]
    # handle uninstalled conda case
    if "bash: conda: command not found" in conda_ret:
        return [], None
    # grab all/current envs
    conda_envs = [line[0] for line in conda_ret]
    conda_env_current = conda_ret["*" in conda_ret][0]
    return conda_envs, conda_env_current

