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
    madminer_install,
    madgraph_install, 
    conda_install, 
    conda_env_install,
    conda_env_activate,
    build_modules,
    installation_directory='',
    run_all=False,
    new_env_name='mm_scripting_util'
):
    """
    creates a bash file which is run for setup. 
    """

    madminer_link="https://github.com/johannbrehmer/madminer.git"
    anaconda_link="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh "
    madgraph_link="https://launchpad.net/mg5amcnlo/2.0/2.6.x/+download/MG5_aMC_v2.6.5.tar.gz"

    module_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # auto install directory: one back from the directory of mm_scripting_util (extremely lame way to do this, I know... sorry)
    if installation_directory is None or len(installation_directory) == 0:
        installation_directory = os.path.abspath(os.path.dirname(module_directory))

    conda_installation_directory = "~"

    if run_all:
        madminer_install = True
        conda_install = True
        conda_env_install = True
        conda_env_activate = True
        build_modules = True

    # conda_envs, current_env = _conda_info()
    
    # open bash file
    with bash_file_wrapper(open("setup_env_util.sh", 'w+')) as f:

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

        if conda_env_activate:
            f.write("echo 'attempting to activate anaconda env'")
            f.write("conda activate {0}".format(new_env_name))
        
        if madminer_install:
            f.write("echo 'attempting to install madminer'")
            if len(madminer_link) > 0:
                f.write("git clone {0} \"{1}/madminer\"".format(madminer_link, installation_directory))
            else:
                f.write("pip install madminer")

        if madgraph_install: 
            f.write("echo 'attempting to install madgraph'")
            f.write("cd ..")
            f.write("wget -c {0}".format(madgraph_link))
            f.write("tar -xzvf MG5_aMC_v2.6.5.tar.gz")
            f.write("rm MG5_aMC_v2.6.5.tar.gz")
            f.write("cd mm_scripting_util")

        # build everything, automatic
        if build_modules:
            f.write("echo 'attempting to build madminer and mm_scripting_util modules'")
            f.write("python setup.py build")
            f.write("cd \"{0}/madminer/\"".format(installation_directory))
            f.write("python setup.py build")
            f.write(" cd \"{0}\"".format(module_directory))

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mm', '--madminer', dest='install_madminer', action='store_true', default=False)
    parser.add_argument('-mg', '--madgraph', dest='install_madgraph', action='store_true', default=False)
    parser.add_argument('-c', '--conda', dest='install_conda', action='store_true', default=False)
    parser.add_argument('-e', '--env-install', dest='install_env', action='store_true', default=False)
    parser.add_argument('-ea', '--activate-env', dest='activate_env', action='store_true')
    parser.add_argument('-ed', '--no-activate-env', dest='activate_env', action='store_false')
    parser.add_argument('-d', '--install-dir', dest='install_directory', action='store', default='', type=str)
    parser.add_argument('-b', '--build', dest='build_modules', action='store_true', default=False)
    parser.add_argument('-a', '--run-all', dest='run_all', action='store_true', default=False)

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        exit(1)
    else:
        print(args)
        args = parser.parse_args(sys.argv[1:])
        write_environment_setup_script(
            args.install_madminer,
            args.install_madgraph,
            args.install_conda,
            args.install_env,
            args.activate_env,
            args.build_modules,
            args.install_directory,
            args.run_all
        )
        exit(0)
    