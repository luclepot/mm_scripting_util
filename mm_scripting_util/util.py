import logging 
import os 
import numpy as np 
import shutil
import platform
import getpass
import traceback
import madminer.core
import madminer.lhe
import madminer.sampling
import madminer.utils.interfaces.madminer_hdf5
import corner
import matplotlib.pyplot as plt
import inspect
import enum 
import collections
import scipy.stats
import tabulate


class mm_base_util():

    """
    base functions and variables used in most other functions. 
    'true' helpers, if you will.  
    Also, baseline init function for dir/name/path etc. 
    """
    
    __CONTAINS_BASE_UTIL = True

    class error_codes(enum.Enum):

        """
        Error code enum class utility; 
        generally helpful for debugging/unit testing. 

        """


        # ones: general errors/success
        
        Success = 0 
        
        Error = 1

        InvalidBackendError = 2
        NoDirectoryError = 3 
        InvalidPlatformError = 4
        InvalidInputError = 5
        InvalidTypeError = 6
        InitError = 7
        CaughtExceptionError = 8
        # tens: simulation errors
        
        NoCardError = 11
        IncorrectCardNumberError = 12
        
        NoMadminerCardError = 13
        
        NoScriptError = 14
        IncorrectScriptNumberError = 15
        
        NoDataFileError = 16
        IncorrectDataFileNumberError = 17

        NoProcessedDataFileError = 18 

        # twenties: training errors

        NoAugmentedDataFileError = 20
        IncorrectAugmentedDataFileError = 21

    Success = error_codes.Success

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
            return self.error_codes.Success
        self.log.error("Init not successful; directory " + self.dir + "does not exist.")
        return self.error_codes.InitError
            
    def _check_ret(
            self, 
            ret,
            ret_warning=None
        ): 
        if self.error_codes.Success not in ret:
            if ret_warning is not None:
                self.log.warning(ret_warning)
            return ret

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

    def _string_find_nth(
            self,
            string,
            substring,
            n
        ):
        parts = string.split(substring, n + 1)
        if len(parts) <= n + 1:
            return -1
        return len(string) - len(parts[-1]) - len(substring)

    def _get_var_name(
            self, 
            var
        ):
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        return [k for k, v in callers_local_vars if v is var]

    def _tabulate_comparison_information(
            self,
            r,
            pers,
            observables, 
            benchmarks,
            threshold=2.0
        ):

        strarr = np.round(np.vstack(r), 2).astype(str)
        index = np.vstack(r) >= threshold
        strarr[index] = np.asarray(["\033[1;37;41m" + elt + "\033[0;37;40m" for elt in strarr[index]])
        columns = ["bin #   "] + ['-\n'.join([benchmark[10*i:10*(i + 1)] for i in range(int(len(benchmark)/10) + 1)]) for obs in observables for benchmark in benchmarks]
        tab_output = tabulate.tabulate(np.vstack([np.vstack([[str(i + 1) for i in range(r.shape[2])], strarr]).T, np.asarray([np.hstack([['failed'], np.asarray(["{:.1%}".format(elt) for elt in np.hstack(pers)])])])]), tablefmt='pipe', headers=columns)
        
        header = " "*(self._string_find_nth(tab_output, '|', 1))

        for i,obs in enumerate(observables):
            header += "| "
            header += obs
            header += " "*(self._string_find_nth(tab_output, '|', len(benchmarks)*(i + 1) + 1) - len(header))
        header += "|"
        
        self.log.info("")
        self.log.info("Printing comparison information for mg5 and augmented histograms...")
        self.log.info("This shows the ratios between the datasets, flagging values for which")
        self.log.info("this ratio is greater than the given threshold of {}.".format(threshold))
        self.log.info("")
        self.log.info(header)
        self.log.info('-'*len(header))
        for line in tab_output.split("\n"):
            self.log.info(line)

    def _exec_wrap(
            self, 
            func
        ):
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except: 
                self.log.error(traceback.format_exc())
        
        return wrapper

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

        if self._check_valid_backend() == self.error_codes.Success:
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
            return self.error_codes.Success

        self.log.warning("Backend found, but parameters were not fully loaded.")        
        # baaad guy error code 
        return self.error_codes.InvalidBackendError

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
            self.log.error("INVALID BACKEND: SEE ERROR LOGS ABOVE")
            return self.error_codes.InvalidBackendError
        return self.error_codes.Success

class mm_simulate_util(
        mm_base_util
    ):
    """
    Container class for simulation-related util functions.
    Seperation from other functions is purely for cleanliness
    while writing. 
    """

    __CONTAINS_SIMULATION_UTIL = True

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
            self._check_valid_mg5_run(samples),
            self._check_valid_mg5_process()
                ]
        while step < len(blist) and blist[step] == self.error_codes.Success:
            step += 1

        return step

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
            return self.error_codes.NoCardError
        if cards != num_cards + 6: 
            self.log.error("Incorrect number of cards in directory " + self.dir)
            self.log.error("expected {}, got {}".format(num_cards + 6, cards))
            return self.error_codes.IncorrectCardNumberError
        return self.error_codes.Success

    def _check_valid_morphing(
            self
        ):
        if self._dir_size(
                pathname=self.dir + "/data",
                matching_pattern="madminer_{}.h5".format(self.name)
            ) != 1:
            self.log.error("More or less than one 'madminer_<name>.h5' card in directory")
            self.log.error("'" + self.dir + "/data'.")
            return self.error_codes.NoMadminerCardError
        return self.error_codes.Success        

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
            return self.error_codes.NoScriptError
        if size != expected:
            self.log.error("Found {}/{} expected mg5 scripts. Incorrect mg5 setup.".format(size, expected))
            return self.error_codes.IncorrectScriptNumberError
        return self.error_codes.Success

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
            return self.error_codes.NoDataFileError
        if size != expected: 
            self.log.error("Found {}/{} expected mg5 data files. Incorrect mg5 setup.".format(size, expected))
            return self.error_codes.IncorrectDataFileNumberError
        return self.error_codes.Success
        
    def _check_valid_mg5_process(
            self    
        ):
        size = self._dir_size(
            self.dir + "/data",
            matching_pattern="madminer_{}_with_data_parton.h5".format(self.name)
        )
        if size < 0:
            self.log.error("/data/ directory does not exist")
            self.log.error("processed mg5 run not completed (or detected)")
            return self.error_codes.NoDirectoryError
        if size == 0:
            self.log.error("No proper processed mg5 file found.")
            self.log.error("processed mg5 run not completed (or detected)")
            return self.error_codes.NoProcessedDataFileError
        return self.error_codes.Success

class mm_train_util(
        mm_base_util
    ):

    __CONTAINS_TRAINING_UTIL = True

    """
    Container class for training-related util functions. 
    Seperation from other functions is just for cleanlines
    while writing. 
    """

    def _get_raw_mg5_arrays(
            self
        ):
        mg5_observations = []
        mg5_weights = []

        for o, w in madminer.utils.interfaces.madminer_hdf5.madminer_event_loader(
            filename=self.dir + "/data/madminer_{}_with_data_parton.h5".format(self.name)
        ):
            mg5_observations.append(o)
            mg5_weights.append(w)

        mg5_obs = np.vstack(np.squeeze(np.asarray(mg5_observations)))
        mg5_weights = np.vstack(np.squeeze(np.asarray(mg5_weights))).T
        mg5_norm_weights = np.copy(mg5_weights) # normalization factors for plots

        n_mg5 = mg5_obs.shape[0]

        self.log.info("correcting normalizations by total sum of weights per benchmark:")

        for i, weight in enumerate(mg5_weights):
            sum_bench = (weight.sum())
            mg5_norm_weights[i] /= sum_bench
            self.log.debug("{}: {}".format(i + 1, sum_bench))
        
        return mg5_obs, mg5_weights, mg5_norm_weights, n_mg5

    def _get_mg5_and_augmented_arrays(
            self,
            training_name,
            bins,            
            ranges,
            dens
        ):  

        rets = [ 
            self._check_valid_augmented_data(training_name=training_name),
            mm_simulate_util._check_valid_mg5_process(self)
            ]
        failed = [ ret for ret in rets if ret != self.error_codes.Success ] 

        if len(failed) > 0:
            self.log.warning("Canceling augmented sampling plots.")            
            return failed, None, None

        # search key for augmented samples
        search_key = "x_{}_augmented_samples_".format(training_name)
        x_files = [f for f in os.listdir(self.dir + "/data/samples") if search_key in f]        
        x_arrays = dict([(f[len(search_key):][:-len(".npy")], np.load(self.dir + "/data/samples/" + f)) for f in x_files])
        # x_size = max([x_arrays[obs].shape[0] for obs in x_arrays])

        n_aug = max([x_arrays[obs].shape[0] for obs in x_arrays])

        # grab benchmarks and observables from files
        (_, 
        benchmarks, 
        _,_,_,
        observables,
        _,_,_,_) = madminer.utils.interfaces.madminer_hdf5.load_madminer_settings(
            filename = self.dir + "/data/madminer_{}_with_data_parton.h5".format(self.name)
        )

        # create lists of each variable
        benchmark_list = [benchmark for benchmark in benchmarks]
        observable_list = [observable for observable in observables]

        mg5_obs, mg5_weights, mg5_norm_weights, n_mg5 = self._get_raw_mg5_arrays()

        x_aug = np.asarray([
            [ 
                np.histogram(
                    x_arrays[benchmark][:,i],
                    bins=bins[i],
                    range=ranges[i],
                    # weights=(mg5_obs[:,i].size/x_arrays[benchmark][:,i].size)*np.ones(x_arrays[benchmark][:,0].shape)*mg5_norm_weights[0][0],
                    density=dens
                )    
                for benchmark in benchmark_list
            ] for i in range(len(observable_list)) 
        ])

        x_mg5 = np.asarray([
            [ 
                np.histogram(
                    mg5_obs[:,i], 
                    range=ranges[i],
                    bins=bins[i],
                    weights=weight,
                    density=dens
                )
                for weight in mg5_norm_weights
            ]    for i in range(len(mg5_obs[0]))
        ])
        aug_values, aug_bins = [np.asarray([[subarr for subarr in arr] for arr in x_aug[:,:,i]]) for i in range(2)]
        mg5_values, mg5_bins = [np.asarray([[subarr for subarr in arr] for arr in x_mg5[:,:,i]]) for i in range(2)]

        digitized = np.asarray([np.digitize(mg5_obs[:,i],np.histogram(mg5_obs[:,i], bins=bins[i] ,range=ranges[i])[1]) for i in range(len(bins)) ]) - 1
        mg5_err = np.asarray([ [ [ np.sqrt(np.sum(weight[digitized[bn] == b]*weight[digitized[bn] == b])) for b in range(bin_n)] for weight in mg5_norm_weights] for bn,bin_n in enumerate(bins)])
        scale_facs = 1./np.asarray(
                [
                    [
                        np.max(
                            np.histogram(
                                mg5_obs[:,i],
                                weights=weight,
                                bins=bins[i],
                                range=ranges[i],
                                density=False
                            )[0]
                        ) / np.max(mg5_values[i,j]) for j, weight in enumerate(mg5_norm_weights)
                    ] for i,obs in enumerate(observables)
                ]
            )
        mg5_err_scaled = (np.vstack(mg5_err).T*np.hstack(scale_facs)).T.reshape(mg5_err.shape)
        
        # xmg5_processed = np.asarray([[subarr for subarr in arr] for arr in x_list_mg5[:,:,0]])

        return [self.error_codes.Success], (aug_values, aug_bins, n_aug), (mg5_values, mg5_bins, n_mg5, mg5_err_scaled)

    def _compare_mg5_and_augmented_data(
            self,
            x_aug, 
            x_mg5,
            y_fac=1.0,
            threshold=2.0
        ):

        r = abs(x_mg5[0] - x_aug[0]) / x_mg5[3]

        pers = [[ len(elt[np.where(elt >= threshold)]) / len(elt) for elt in relt] for relt in r]
        
        return r, pers

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
            return self.error_codes.NoDirectoryError
        elif size == 0:
            self.log.error("No training data to parse.")
            return self.error_codes.NoDataFileError
        return self.error_codes.Success

    def _check_valid_augmented_data(
            self,
            training_name,
            expected_benchmarks=None
        ):
        size = self._dir_size(
            pathname=self.dir + '/data/samples',
            matching_pattern=training_name + "_augmented_samples"
        )

        if size < 0:
            self.log.error("data/samples directory does not exist!")
            self.log.error("Augmented data not parsed (or detected)")
            return self.error_codes.NoDirectoryError
        elif size == 0:
            self.log.error("data/samples directory does not contain any files with the given training_name")
            self.log.error("Augmented data not parsed (or detected)")
            return self.error_codes.NoAugmentedDataFileError
        ret = self.error_codes.Success
        if expected_benchmarks is not None:
            if size != expected_benchmarks:
                self.log.error("data/samples directory contains an incorrect # of files with given training_name ")
                self.log.error(" - {}/{} expected files".format(size, expected_benchmarks)) 
                ret = self.error_codes.IncorrectAugmentedDataFileError

        files = [f for f in os.listdir(self.dir + '/data/samples') if (training_name + "_augmented_samples") in f]
        self.log.debug("Files found:")
        for f in files: 
            self.log.debug(f)
        return ret
        # return self.error_codes.Success

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
    
    def _submit_condor(
            self,
            arg_list,
            max_runtime=60*60
        ):

        # write condor submission file
        with open("{}/{}.sub".format(self.path, self.name), 'w+') as f:
            f.write("executable = python\n")
            f.writelines("arguments = -m mm_scripting_util {}\n".format(" ".join(arg_list)))
            f.write("output = {}/output/{}.output.$(ClusterId).$(ProcId).out\n".format(self.path, self.name))
            f.write("error = {}/error/{}.error.$(ClusterId).$(ProcId).err\n".format(self.path, self.name))
            f.write("log = {}/log/{}.log.$(ClusterId).$(ProcId).log\n".format(self.path, self.name))
            f.write("+MaxRuntime = {}\n".format(max_runtime))
            f.write("queue\n")
        # variable_string =   "MM_NAME=\"{}\";".format(self.name) + \
        #                     "MM_MAX_RUNTIME={};".format(max_runtime) + \
        #                     "MM_RUN_DIR=\"{}\";".format(self.dir) + \
        #                     "MM_MOD_DIR=\"{}/data/condor\";".format(self.module_path) + \
        #                     "MM_ARG_STR=\"-m mm_scripting_util.run {}\";".format(" ".join(arg_list)) + \
        #                     "echo $MM_NAME; echo $MM_MAX_RUNTIME; echo $MM_RUN_DIR; echo $MM_MOD_DIR; echo $MM_ARG_STR;" 
        #                     # "condor_submit {}/data/condor/core.sub".format(self.module_path)

        os.system("condor_submit {}/{}.sub".format(self.path, self.name))

        return self.error_codes.Success

    
    def _submit_flashy(
            self,
            arg_list,
            max_runtime=60*60 # 1 hour
        ):

        return self.error_codes.Success