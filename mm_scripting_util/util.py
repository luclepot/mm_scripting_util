import logging 
import os 
import numpy as np 
import shutil
import platform
import getpass
import traceback

class tth_base_util:

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
        self.log.warning("Init not successful; directory " + self.dir + "does not exist.")
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

class tth_simulate_util(
        tth_base_util
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
            self.log.warning("No valid cards directory in " + self.dir)
            return False
        if cards != num_cards + 6: 
            self.log.warning("Incorrect number of cards in directory " + self.dir)
            self.log.warning("expected {}, got {}".format(num_cards + 6, cards))
            return False
        return True

    def _check_valid_morphing(
            self
        ):
        if self._dir_size(
                pathname=self.dir + "/data",
                matching_pattern="madminer_example.h5"
            ) != 1:
            self.log.warning("More or less than one 'madminer_example.h5' card in directory")
            self.log.warning("'" + self.dir + "/data'.")
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
        expected = len(self._equal_sample_sizes(
            samples=samples,
            sample_limit=100000
        ))
        if size < 0:
            self.log.warning("mg_processes/signal/madminer/scripts directory does not exist here.")
            return False
        if size != expected:
            self.log.warning("Found {}/{} expected mg5 scripts. Incorrect mg5 setup.".format(size, expected))
            return False
        return True

    def _check_valid_mg5_run(
            self,
            samples
        ):
        expected = len(self._equal_sample_sizes(
            samples=samples, 
            sample_limit=100000
        ))
        size = self._dir_size(
            pathname=self.dir + '/mg_processes/signal/Events', 
            matching_pattern="run_"
        )
        if size < 0:
            self.log.warning("mg_processes/signal/Events directory does not exist!")
            self.log.warning("mg5 run not completed (or detected)")
            return False
        if size != expected: 
            self.log.warning("Found {}/{} expected mg5 data files. Incorrect mg5 setup.".format(size, expected))
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

class tth_train_util(
        tth_base_util
    ):

    __CONTAINS_TRAINING_UTIL = True

    """
    Container class for training-related util functions. 
    Seperation from other functions is just for cleanlines
    while writing. 
    """
    pass

class tth_util(
        tth_simulate_util,
        tth_train_util,
        tth_base_util
    ):
    """
    Wrapper class for all tth utility related classes. 
    Combines simulation, training, and baseline utility classes in one import class. 
    From this we will derive our tth_miner class, with first-order functions. 
    """
    pass 