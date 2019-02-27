import logging 
import os 
import numpy as np 
import shutil
import platform
import getpass
import traceback
import madminer.core
import madminer.lhe
import corner
import matplotlib.pyplot as plt

class mm_base_util:

    """
    base functions and variables used in most other functions. 
    'true' helpers, if you will.  
    Also, baseline init function for dir/name/path etc. 
    """
    
    __CONTAINS_BASE_UTIL = True

    def __init__(
            self,
            name,
            path
        ):
        self.name = name
        self.path = path
        self.dir = self.path + "/" + self.name
        self.log = logging.getLogger(__name__)
        self.module_path = os.path.dirname(__file__)

    def _check_valid_init(
            self
        ):
        if os.path.exists(self.dir):
            return True
        self.log.error("Init not successful; directory " + self.dir + "does not exist.")
        return False
            
    def _dir_size(
            self, 
            pathname,
            matching_pattern=""
        ):
        """Description:
            Helper function returning number of files in pathname matching the given pattern.
            Warns if matching pattern is 
        
        Parameters: 
            pathname
                string. sampling directory to search. 
            matching_pattern
                string, default empty. file pattern to check files against. 

        Returns: 
            int, number of matching files in directory, or -1 if dir does not exist.
        """

        if os.path.exists(pathname):
            dir_elts = len([elt for elt in os.listdir(pathname) if matching_pattern in elt])
            return dir_elts
        return -1 

    def _replace_lines(
            self, 
            infile, 
            line_numbers, 
            line_strings, 
            outfile=None         
        ):
        if outfile is None:
            outfile = infile
        assert(len(line_numbers) == len(line_strings))
        with open(infile, 'r') as file:
            lines = file.readlines()

        for i, line_number in enumerate(line_numbers):
            assert(line_number <= len(lines))
            lines[line_number - 1] = line_strings[i]

        with open(outfile, 'w+') as file:
            file.writelines(lines)

    def _remove_files(
            self, 
            path_to_clear,
            include_folder=False,
            matching_pattern=""
        ):
        if os.path.exists(path_to_clear):
            if include_folder and matching_pattern is "": 
                if platform.system() == "Linux": 
                    if include_folder: 
                        cmd = "rm -r \"{}\"".format(path_to_clear)
                    else:
                        cmd = "rm -r \"{}\"/*".format(path_to_clear)
                elif platform.system() == "Windows":
                    if include_folder: 
                        cmd = "rmdir /s \"{}\"".format(path_to_clear)
                    else:
                        cmd = "rmdir /s /q \"{}\"\\".format(path_to_clear)
                else:
                    return
                os.system(cmd)
                self.log.info("Removed directory {}".format(path_to_clear))
            else: 
                matching_files = [elt for elt in os.listdir(path_to_clear) if matching_pattern in elt]
                for f in matching_files: 
                    os.remove(path_to_clear + "/" + f)
                    self.log.debug("Removing file {}/{}".format(path_to_clear, f))
                self.log.info("Removed files from directory {}".format(path_to_clear))

    def _check_directory(
            self,
            local_pathname,
            force,
            pattern="",
            mkdir_if_not_existing=True
        ):
        
        dirsize = self._dir_size(
            pathname=self.dir + "/" + local_pathname,
            matching_pattern=pattern
            )

        if dirsize < 0:
            if mkdir_if_not_existing: 
                os.mkdir(self.dir + "/" + local_pathname)
            else:
                return

        elif dirsize > 0:
            self.log.warning("existing data in specified directory \"{}\"".format(local_pathname)) 
            if not force:
                raise FileExistsError("Directory {} not empty of data".format(local_pathname))
            else:
                self.log.info("Force flag triggered. Removing data in directory \"{}\".".format(local_pathname))
                self._remove_files(
                    self.dir + "/" + local_pathname,
                    include_folder=False,
                    matching_pattern=pattern
                    )

    def _search_for_paths(
            self,
            pathname,
            include_module_paths=True   
        ):
        # search first for exact path

        if pathname is None:
            return None
        
        if os.path.exists(pathname):
            ret = pathname 

        # then, search for local pathnames (raw and /data/)
        elif os.path.exists(self.dir + "/" + pathname):
            ret = self.dir + pathname
        elif os.path.exists(self.dir + "/data/" + pathname):
            ret = self.dir + "/data/" + pathname
        # check for terminal cwd paths
        elif os.path.exists(os.getcwd() + "/" + pathname):
            ret = os.getcwd() + "/" + pathname
        
        # last, check default file database (data/cards/, data/backends/, data/, and raw)
        elif include_module_paths and os.path.exists(self.module_path + "/data/backends/" + pathname):
            ret = self.module_path + "/data/backends/" + pathname
        elif include_module_paths and os.path.exists(self.module_path + "/data/" + pathname):
            ret = self.module_path + "/data/" + pathname
        elif include_module_paths and os.path.exists(self.module_path + "/data/cards/" + pathname):
            self.module_path + "/data/cards/" + pathname
        elif include_module_paths and os.path.exists(self.module_path + "/" + pathname):
            ret = self.module_path + "/" + pathname
        else: 
            self.log.error("Could not find pathname {}".format(pathname))
            return None
        return ret
        # otherwise this doesn't exist

class mm_backend_util(
        mm_base_util
    ):

    __CONTAINS_BACKEND_UTIL = True

    def __init__(
            self,
            required_params=["model", "madgraph_generation_command", "backend_name", "parameters", "benchmarks", "observables"],
            required_experimental_params=["lha_block", "lha_id", "morphing_max_power", "parameter_range"]
        ):
        self.params = {}
        # default required: model type, madgraph generation command, and backend name
        self.required_params = required_params
        self.required_experimental_params = required_experimental_params

    def _load_backend(
            self,
            backend
        ):

        self.backend_name = self._search_for_paths(pathname=backend)    
        if self.backend_name is None:
            self.log.warning("No backend found, and none loaded.")
            return 1

        # process backend file
        with open(self.backend_name) as f:
            
            self.log.info("Attempting to load backend file at {}".format(self.backend_name))

            parameters = {}
            benchmarks = {}
            observables = {}

            for l in f:

                line = l.lstrip().rstrip("\n")

                # if resultant line is not empty, and is not a comment
                if len(line) > 0 and line[0] != "#":

                    # parameter read-in case
                    if "parameter " in line: 
                        name, parameter_dict = self._get_parameter_dict(line)
                        parameters[name] = parameter_dict

                    # benchmark read-in case
                    elif "benchmark " in line: 
                        name, benchmark_dict = self._get_benchmark_dict(line)
                        benchmarks[name] = benchmark_dict

                    # observable read-in case
                    elif "observable " in line: 
                        name, observable_dict = self._get_obseravble_dict(line)
                        observables[name] = observable_dict

                    # otherwise, default case (add that string to the dictionary)
                    else: 
                        line = line.split(": ")
                        self.params[line[0].lstrip().rstrip()] = line[1].lstrip().rstrip()

            # add filled dictionaries in their respective places
            self.params["parameters"] = dict([(p,parameters[p]) for p in parameters if p is not None])
            self.params["benchmarks"] = benchmarks
            self.params["observables"] = observables

        # verify required backend parameters in backend file

        if self._check_valid_backend():
            self.log.info("Loaded {} parameters for backend with name {}".format(len(self.params), self.params["backend_name"]))
            self.log.debug("")
            self.log.debug("--- Backend parameter specifications ---")
            self.log.debug("")
            for required_parameter in self.required_params:
                self.log.debug(required_parameter + ": ")
                if type(self.params[required_parameter]) == dict:
                    for subparam in self.params[required_parameter]:
                        if type(self.params[required_parameter][subparam]) == dict: 
                            self.log.debug("  - {}:".format(subparam))
                            for subsubparam in self.params[required_parameter][subparam]:
                                self.log.debug("    - {}: {}".format(subsubparam, self.params[required_parameter][subparam][subsubparam]))
                        else:
                            self.log.debug("  - {}: {}".format(subparam, self.params[required_parameter][subparam]))
                else:
                    self.log.debug("  - {}".format(self.params[required_parameter]))
            self.log.debug("")
            return 0

        self.log.warning("Backend found, but parameters were not fully loaded.")        
        # baaad guy error code 
        return 1

    def _get_parameter_dict(
            self, 
            line
        ):
        og = line
        line = line.split(": ")
        name = line[0].lstrip("parameter ")
        line = line[1]
        try:
            linedict = {}
            line = [value.lstrip().rstrip() for value in line.split(", ")]
            for value in line:
                value = value.split("=")
                # convert to an int
                if value[0] in ["lha_id", "morphing_max_power"]:
                    value[1] = int(value[1])
                    # convert to a tuple
                elif value[0] in ["parameter_range"]:
                    value[1] = tuple([float(v) for v in value[1].lstrip("(").rstrip(")").split(",")])
                # elif value[0] in ["parameter_benchmarks"]:
                #     value[1] = [tuple((float(tup[0]), str(tup[1].lstrip("'").rstrip("'")))) for tup in [elt.lstrip("[(").rstrip(")]").split(",") for elt in value[1].split("),(")]]
                elif value[0] not in self.required_experimental_params:
                    self.log.error("invalid parameter argument '{}'.".format(value[0]))
                    return (None,None)
                linedict[value[0]] = value[1]
            return (name,linedict)
        
        except Exception as e:
            self.log.error("incorrect parameter formatting for parameter {}".format(name))
            self.log.error("'{}' not formatted quite right".format(og))
            self.log.error("Full exception:")
            self.log.error(e)
            return (None,None)

    def _get_benchmark_dict(
            self,
            string
        ): 
        string = string.split(": ")
        name = string[0].lstrip("benchmark ")
        tags = dict([tuple((elt.split("=")[0], float(elt.split("=")[1]))) for elt in string[1].rstrip().lstrip().split(", ")])
        return name, tags

    def _get_obseravble_dict(
            self, 
            string
        ):
        string = string.split(": ")
        name = string[0].lstrip("observable ")
        return name, string[1]

    def _check_valid_backend(
            self
        ):

        valid_flag = True

        for param in self.required_params:
            if param not in self.params:
                self.log.error("Provided backend file does not include the required key '{}'".format(param))
                valid_flag = False

        if not len(self.params["parameters"]) > 0:
            self.log.error("Zero parameters provided in backend file. Please specify.")
            valid_flag = False
        for exp_param in self.params["parameters"]:
            for req_exp_param in self.required_experimental_params:
                if req_exp_param not in self.params["parameters"][exp_param]:
                    self.log.error("Required experimental parameter '{}' not in parameter '{}'".format(req_exp_param, exp_param))
                    valid_flag = False

        for benchmark in self.params["benchmarks"]: 
            for existing_parameter in self.params["parameters"]:
                if existing_parameter not in self.params["benchmarks"][benchmark]:
                    self.log.error("Provided sample benchmark '{}' does not contain a value for parameter '{}'.".format(self.params["benchmarks"][benchmark], existing_parameter))
                    valid_flag = False            

        if not valid_flag: 
            # self.log.error("Please update backend file '{}'".format(self.backend_name))
            return False
        return True

class mm_simulate_util(
        mm_base_util
    ):
    """
    Container class for simulation-related util functions.
    Seperation from other functions is purely for cleanliness
    while writing. 
    """

    __CONTAINS_SIMULATION_UTIL = True

    def _check_valid_cards(
            self,
            num_cards
        ):
        """Description:
            Helper function to check validity of cards. 

        Parameters: 
            num_cards
                int. expected number of run_cards (the rest are accounted to). 

        Returns: 
            bool. number of cards in directory, or -1 if card dir does not exist.
        """
        cards = self._dir_size(
            pathname=self.dir + "/cards",
            matching_pattern="card"
        )
        if cards < 0: 
            self.log.error("No valid cards directory in " + self.dir)
            return False
        if cards != num_cards + 6: 
            self.log.error("Incorrect number of cards in directory " + self.dir)
            self.log.error("expected {}, got {}".format(num_cards + 6, cards))
            return False
        return True

    def _check_valid_morphing(
            self
        ):
        if self._dir_size(
                pathname=self.dir + "/data",
                matching_pattern="madminer_example.h5"
            ) != 1:
            self.log.error("More or less than one 'madminer_example.h5' card in directory")
            self.log.error("'" + self.dir + "/data'.")
            return False
        return True        

    def _check_valid_mg5_scripts(
            self,
            samples
        ):
        size = self._dir_size(
            pathname=self.dir + '/mg_processes/signal/madminer/scripts',
            matching_pattern=".sh",
        )
        expected = self._number_of_cards(
            samples=samples,
            sample_limit=100000
        )
        if size < 0:
            self.log.error("mg_processes/signal/madminer/scripts directory does not exist here.")
            return False
        if size != expected:
            self.log.error("Found {}/{} expected mg5 scripts. Incorrect mg5 setup.".format(size, expected))
            return False
        return True

    def _check_valid_mg5_run(
            self,
            samples
        ):
        expected = self._number_of_cards(
            samples=samples, 
            sample_limit=100000
        )
        size = self._dir_size(
            pathname=self.dir + '/mg_processes/signal/Events', 
            matching_pattern="run_"
        )
        if size < 0:
            self.log.error("mg_processes/signal/Events directory does not exist!")
            self.log.error("mg5 run not completed (or detected)")
            return False
        if size != expected: 
            self.log.error("Found {}/{} expected mg5 data files. Incorrect mg5 setup.".format(size, expected))
            return False
        return True
        
    def _check_valid_mg5_process(
            self    
        ):
        size = self._dir_size(
            self.dir + "/data",
            matching_pattern="madminer_example_with_data_parton.h5"
        )
        if size < 0:
            self.log.error("/data/ directory does not exist")
            self.log.error("processed mg5 run not completed (or detected)")
            return False
        if size == 0:
            self.log.error("No proper processed mg5 file found.")
            self.log.error("processed mg5 run not completed (or detected)")
            return False
        return True

    def _equal_sample_sizes(
            self, 
            samples,
            sample_limit
        ):
        sample_sizes = [sample_limit for i in range(int(samples/sample_limit))]
        
        if int(samples % int(sample_limit)):
            sample_sizes += [int(samples % int(sample_limit))]
        
        return sample_sizes

    def _number_of_cards(
            self,
            samples, 
            sample_limit            
        ):
        size = int(samples/sample_limit)
        if int(samples % int(sample_limit)):
            return size + 1
        return size

    def _get_simulation_step(
            self, 
            num_cards,
            samples
        ):
        step = 0
        blist = [
            self._check_valid_cards(num_cards),
            self._check_valid_morphing(),
            self._check_valid_mg5_scripts(samples),
            self._check_valid_mg5_run(samples)
                ]
        while step < len(blist) and blist[step]:
            step += 1

        return step

class mm_train_util(
        mm_base_util
    ):

    __CONTAINS_TRAINING_UTIL = True

    """
    Container class for training-related util functions. 
    Seperation from other functions is just for cleanlines
    while writing. 
    """
    
    def _check_valid_training_data(
            self
        ):

        size = self._dir_size(
            pathname=self.dir + '/mg_processes/signal/Events', 
            matching_pattern="run_"
        )
        if size < 0:
            self.log.error("mg_processes/signal/Events directory does not exist!")
            self.log.error("Training data cannot be parsed (or detected)")
            return False
        return True

class mm_util(
        mm_backend_util,
        mm_simulate_util,
        mm_train_util,
        mm_base_util
    ):
    """
    Wrapper class for all tth utility related classes. 
    Combines simulation, training, and baseline utility classes in one import class. 
    From this we will derive our tth_miner class, with first-order functions. 
    """
    pass 