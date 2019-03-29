import os
import sys
import subprocess 
import argparse

def _conda_info(
):
    """
    Returns the current os' installed conda environments, as well as the current env
    Returns empty list, None if conda is not installed.
    """

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

class bash_file_wrapper:
    def __init__(
        self, 
        opened_file_object
    ):
        self.f = opened_file_object
        self.write("#!/bin/bash")

    def __enter__(
        self
    ): 
        return self

    def __exit__(
        self,
        exc_type, 
        exc_val,
        exc_tb
    ):
        self.f.close()

    def write(
        self,
        to_write=''
    ):
        if type(to_write) == list:
            for wstr in to_write: 
                self.write(wstr)
        else:
            self.f.write(str(to_write) + "\n")
    
    def write_raw(
        self, 
        to_write=''
    ):
        self.f.write(to_write)

def write_environment_setup_script(
    include_madgraph_install=False,
    installation_directory=None,
    new_env_name="mm_scripting_util",
    madminer_repository="git@github.com:johannbrehmer/madminer.git"
):
    """
    creates a bash file which is run for setup. 
    """

    module_directory = os.path.abspath(__file__)

    # auto install directory: one back from the directory of mm_scripting_util (extremely lame way to do this, I know... sorry)
    if installation_directory is None:
        installation_directory = os.path.dirname(module_directory)

    conda_envs, current_env = _conda_info()

    # open bash file
    with bash_file_wrapper(open("setup_env_util.sh", 'w+')) as f:

        # if no conda installed/activated... 
        if current_env is None:
            f.write("# install anaconda")
            f.write("wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O \"{0}/miniconda.sh\"".format(installation_directory))
            f.write("bash \"{0}/miniconda.sh\" -b -p \"{1}/miniconda\"".format(installation_directory, installation_directory))
            f.write("source \"{0}/miniconda/etc/profile.d/conda.sh\"".format(installation_directory))
            f.write("rm \"{0}/miniconda.sh\"".format(installation_directory))
            f.write()

        # if the required environment doesn't exist, create it
        if "mm_scripting_util" not in conda_envs:
            f.write("# create anaconda environment from file")
            f.write("conda env create -n {0} -f \"{1}/environment.yml\"".format(new_env_name, module_directory))

        f.write("# activate anaconda env")
        f.write("conda activate {0}".format(new_env_name))
        f.write("# install madminer, build")
        f.write("git clone {0} \"{1}/madminer\"".format(madminer_repository, installation_directory))
        f.write("python \"{0}/madminer/setup.py build\"".format(installation_directory))
        f.write("")

        if include_madgraph_install: 
            f.write("# install madgraph dir")
            f.write("wget -qO- https://launchpad.net/mg5amcnlo/2.0/2.6.x/+download/MG5_aMC_v2.6.5.tar.gz | tar xvz - -C \"{0}\"".format(installation_directory))

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--include-madgraph', dest='include_madgraph',
        action='store_true')
    args = parser.parse_args(sys.argv)
    write_environment_setup_script(include_madgraph_install=args.include_madgraph)
    