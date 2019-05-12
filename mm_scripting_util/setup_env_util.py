import os
import sys
import subprocess 
import argparse
import glob

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

class bash_file_wrapper(
):
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
    madgraph_install, 
    conda_install, 
    conda_env_install,
    build_modules,
    installation_directory='',
    run_all=False,
    setup_alias=True,
    new_env_name='mm_scripting_util'
):
    """
    creates a bash file which is run for setup. 
    """

    anaconda_link="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh "
    madgraph_link="https://launchpad.net/mg5amcnlo/2.0/2.6.x/+download/MG5_aMC_v2.6.5.tar.gz"

    module_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # auto install directory: one back from the directory of mm_scripting_util (extremely lame way to do this, I know... sorry)
    if installation_directory is None or len(installation_directory) == 0:
        installation_directory = os.path.abspath(os.path.dirname(os.path.abspath(module_directory)))

    conda_installation_directory = "~"

    if run_all:
        madminer_install = True
        conda_install = True
        conda_env_install = True
        build_modules = True
        setup_alias = True

    # conda_envs, current_env = _conda_info()
    
    # open bash file
    with bash_file_wrapper(open("setup_env_util.sh", 'w+')) as f:

        if setup_alias:
            f.write("alias mm_scripting_util='python \"{0}/run.py\" \"$@\"'".format(os.path.dirname(os.path.abspath(__file__))))
            f.write("alias mmsc='python \"{0}/run.py\" \"$@\"'".format(os.path.dirname(os.path.abspath(__file__))))

        if conda_install:
            f.write("echo 'attempting to install anaconda..'")
            f.write("wget -nv {0} -O miniconda.sh".format(anaconda_link))
            # f.write("echo 'got miniconda source?'")
            f.write("chmod +x miniconda.sh")
            f.write("bash miniconda.sh -b -p {0}/miniconda".format(conda_installation_directory))
            f.write("source {0}/miniconda/etc/profile.d/conda.sh".format(conda_installation_directory))
            # f.write("echo 'source {0}/miniconda/etc/profile.d/conda.sh' >> ~/.bashrc".format(conda_installation_directory))
            f.write("rm miniconda.sh")

        if conda_env_install:
            f.write("echo 'attempting to create anaconda environment from file'")
            f.write("conda env create -n {0} -f \"{1}/environment.yml\"".format(new_env_name, module_directory))
            f.write("conda activate {0}".format(new_env_name))
        
        # if madminer_install:
        #     # f.write("echo 'attempting to install madminer'")
        #     # if len(madminer_link) > 0:
        #     #     f.write("git clone {0} \"{1}/madminer\"".format(madminer_link, installation_directory))
        #     f.write("pip install madminer")

        if madgraph_install:
            f.write("echo 'attempting to install madgraph'")
            f.write("cd ..")
            f.write("wget -c {0}".format(madgraph_link))
            f.write("tar -xzvf MG5_aMC_v2*.tar.gz > mginstallout.txt")
            f.write("rm mginstallout.txt")
            f.write("rm MG5_aMC_v2*.tar.gz")
            f.write("cd mm_scripting_util")

        # build everything, automatic
        if build_modules:
            f.write("echo 'attempting to build madminer and mm_scripting_util modules'")
            f.write("python setup.py develop")
            # f.write("cd \"{0}/madminer/\"".format(installation_directory))
            # f.write("python setup.py develop")
            # f.write(" cd \"{0}\"".format(module_directory))

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mg', '--madgraph', dest='install_madgraph', action='store_true', default=False, help='download and install the latest version of madgraph to the home directory (or install-dir if specified)')
    parser.add_argument('-c', '--conda', dest='install_conda', action='store_true', default=False, help='installs miniconda to the home directory (or install-dir if specified)')
    parser.add_argument('-e', '--env', dest='install_env', action='store_true', default=False, help='installs and activates a conda environment supporting this module')
    parser.add_argument('-d', '--dir', dest='install_directory', action='store', default='', type=str, help='directory to put all installed objects')
    parser.add_argument('-b', '--build', dest='build_modules', action='store_true', default=False, help='attempts to build this module')
    parser.add_argument('-a', '--alias', dest='write_alias', action='store_true', default=False, help='associates an alias in this current shell such that running `mmsc` or `mm_scripting_util` is equivalent to python mm_scripting_util/run.py')
    parser.add_argument('-all', '--all', dest='run_all', action='store_true', default=False, help='runs all of the above commands with the exception installing madgraph.')

    if len(sys.argv[1:]) == 0:
        write_environment_setup_script(
            False,
            False,
            False, 
            False, 
            False,
            False,
            False,
        )
        exit(1)
    else:
        args = parser.parse_args(sys.argv[1:])
        write_environment_setup_script(
            args.install_madgraph,
            args.install_conda,
            args.install_env,
            args.build_modules,
            args.install_directory,
            args.run_all,
            args.write_alias,
        )
        exit(0)
    