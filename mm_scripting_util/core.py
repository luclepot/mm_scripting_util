
from .util import *

class miner(mm_util):
    madminer.core
    """
    Main container for the class. 

    This class should handle all necessary interfacing with
    madminer and madgraph. Nice! 

    """

    __MODULE_NAME = "mm_scripting_util"

    # general class member functions

    def __init__(
        self,
        name="temp",
        path=None,
        loglevel=logging.INFO,
        madminer_loglevel=logging.INFO,
        autodestruct=False,
        backend="tth.dat",
        custom_card_directory=None
    ):
        """
        madminer-helper object for quickly running madgraph scripts. 
        parameters:
            name:
                string, the name of the main object directory
            path: 
                string, path to desired object directory. Will default to current working directory.
            loglevel:
                int (enum), loglevel for tth class as a whole. Will default to INFO.
            autodestruct:
                bool, cleanup switch for object// all files. if true, the whole thing will destroy itself upon deletion. useful for testing.
            backend:
                string, path to a backend file. Examples found at "mm_scripting_util/data/backends/". Provides all benchmarks/ simulation information
            custom_card_directory:
                string, path to a card directory from which to load template cards, if one desires to switch the current template cards out for new ones.
        """

        if path is None:
            path = os.getcwd()
        
        # initialize helper classes
        
        mm_base_util.__init__(
                self,
                name,
                path
            )

        mm_backend_util.__init__(
                self
            )

        self.autodestruct = autodestruct
        self.log = logging.getLogger(__name__)
        self.module_path = os.path.dirname(__file__)

        self.set_loglevel(loglevel)
        self.set_loglevel(madminer_loglevel, module="madminer")

        self.name = name
        self.dir = "{}/{}".format(self.path, self.name)

        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
            self.log.info("Creating new directory " + self.dir)
        else: 
            self.log.info("Initialized to existing directory " + self.dir)

        self.log.debug("Initialized a new miner object with name '{}'".format(self.name))
        self.log.debug("- module path at {}".format(self.module_path))
        self.log.debug("- new miner object path at " + self.dir)

        self.madminer_object = madminer.core.MadMiner()
        self.lhe_processor_object = None
        
        self.log.debug("Loading custom card directory... ")
        if custom_card_directory is not None:
            ret = self._search_for_paths(custom_card_directory, include_module_paths=False)
            if ret is None: 
                self.log.error("Selected custom card directory '{}' could not be found.".format(custom_card_directory))
                self.log.error("Using default card directory instead.")
                self.custom_card_directory = None
            else:
                self.log.debug("Using custom card directory '{}'".format(ret))
                self.custom_card_directory = ret
        else: 
            self.custom_card_directory = None
            self.log.debug("No custom card directory provided.")

        self._load_backend(backend)

    def set_loglevel(
        self,
        loglevel,
        module=None
    ):     

        logging.basicConfig(
            format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
            datefmt='%H:%M',
            level=logging.WARNING
        )

        if module is None:
            module = self.__MODULE_NAME

        logging.getLogger(module).setLevel(loglevel)
        
        return loglevel 

    def destroy_sample(
        self
    ):

        rets = [
            self._check_valid_init()
            ]
        failed = [ret for ret in rets if ret != self.error_codes.Success]

        if len(failed) > 0:
            return failed

        self._remove_files(
                self.dir,
                include_folder=True
            )

        return [self.error_codes.Success]
    
    def list_augmented_samples(
        self,
        verbose=True
    ):
        self.log.info("Augmented samples:")
        samples = []
        sample_list = glob.glob("{}/data/samples/*/augmented_sample.mmconfig".format(self.dir))
        for i,sample in enumerate(sample_list):
            samples.append((sample, self._load_config(sample)))
            self.log.info(" - {}".format(samples[i][1]['sample_name']))
            if verbose:
                for elt in samples[i][1]:
                    if elt is not 'sample_name':
                        self.log.info("    - {}: {}".format(elt, samples[i][1][elt]))
        return samples

    def list_trained_models(
        self,
        verbose=True
    ):
        models = []
        model_list = glob.glob("{}/models/*/*/training_model.mmconfig".format(self.dir))

        self.log.info("Trained Models:")

        for i,model in enumerate(model_list):
            models.append((model, self._load_config(model)))
            self.log.info(" - {}".format(models[i][1]['training_name']))
            if verbose:
                for elt in models[i][1]:
                    if elt is not 'training_name':
                        self.log.info("    - {}: {}".format(elt, models[i][1][elt]))
        return models

    def list_evaluations(
        self,
        verbose=True
    ):
        evaluations = [ ]
        evaluation_list = glob.glob("{}/evaluations/*/*/evaluation.mmconfig".format(self.dir))
        for i,evaluation in enumerate(evaluation_list):
            evaluations.append((evaluation, self._load_config(evaluation)))
            self.log.info(" - {}".format(evaluations[i][1]['evaluation_name']))
            if verbose: 
                for elt in evaluations[i][1]:
                    self.log.info("    - {}: {}".format(elt, evaluations[i][1][elt]))
        return evaluations
    
    def __del__(
        self
    ):
        if self.autodestruct:
            self.destroy_sample()

    # simulation-related member functions
    
    def simulate_data(
        self,
        samples,
        sample_benchmark,
        seed_file=None, 
        force=True,
        mg_dir=None,
        use_pythia_card=False,
        platform="lxplus7",
        morphing_trials=2500,
        override_step=None
    ):
        """
        Standard data simulation run. Should go from start to finish with data simulation.
        """
        try:
            if override_step is not None:
                self.SIMULATION_STEP = override_step
            else: 
                self.SIMULATION_STEP = self._get_simulation_step(
                        self._number_of_cards(samples, 100000),
                        samples
                    )

            if self.SIMULATION_STEP < 1 or force:
                self.log.debug("")
                self.log.debug("RUNNING SETUP CARDS, STEP 1")
                self.log.debug("")
                ret = self.setup_cards(
                        n_samples=samples,
                        seed_file=seed_file,
                        force=force
                    )
                if self.error_codes.Success not in ret:
                    self.log.warning("Quitting simulation with errors.")
                    return ret
                self.SIMULATION_STEP = 1
                self.log.debug("")
                self.log.debug("FINISHED SETUP CARDS, STEP 1")
                self.log.debug("")

            if self.SIMULATION_STEP < 2 or force:
                self.log.debug("")
                self.log.debug("RUNNING MORPHING, STEP 2")
                self.log.debug("")
                ret = self.run_morphing(
                        force=force,
                        morphing_trials=morphing_trials
                    )
                if self.error_codes.Success not in ret:
                    self.log.warning("Quitting simulation with errors.")
                    return ret
                self.SIMULATION_STEP = 2
                self.log.debug("")
                self.log.debug("FINISHED MORPHING, STEP 2")
                self.log.debug("")

            if self.SIMULATION_STEP < 3 or force:   
                self.log.debug("")
                self.log.debug("RUNNING SETUP MG5 SCRIPTS, STEP 3")
                self.log.debug("")
                ret = self.setup_mg5_scripts(
                        samples=samples,
                        sample_benchmark=sample_benchmark,
                        force=force,
                        mg_dir=mg_dir,
                        platform=platform,
                        use_pythia_card=use_pythia_card,
                    )
                if self.error_codes.Success not in ret:
                    self.log.warning("Quitting simulation with errors.")
                    return ret
                self.SIMULATION_STEP = 3
                self.log.debug("")
                self.log.debug("FINISHED SETUP MG5 SCRIPTS, STEP 3")
                self.log.debug("")

            if self.SIMULATION_STEP < 4 or force:        
                self.log.debug("")
                self.log.debug("RUNNING MG5 SCRIPTS, STEP 4")
                self.log.debug("")
                ret = self.run_mg5_script(
                        platform=platform,
                        samples=samples,
                        force=force
                    )
                if self.error_codes.Success not in ret:
                    self.log.warning("Quitting simulation with errors.")
                    return ret
                self.SIMULATION_STEP = 4
                self.log.debug("")
                self.log.debug("FINISHED MG5 SCRIPTS, STEP 4")
                self.log.debug("")

            if self.SIMULATION_STEP < 5 or force:
                self.log.debug("")
                self.log.debug("RUNNING MG5 DATA PROCESS, STEP 5")
                self.log.debug("")
                ret = self.process_mg5_data()
                if self.error_codes.Success not in ret:
                    self.log.warning("Quitting simulation with errors.")
                    return ret
                self.SIMULATION_STEP = 5
                self.log.debug("")
                self.log.debug("FINISHED MG5 DATA PROCESS, STEP 5")
                self.log.debug("")

        except:
            self.log.error(traceback.format_exc())
            self.log.error("ABORTING")
            return [self.error_codes.CaughtExceptionError]

        return [self.error_codes.Success]

    def setup_cards(
        self, 
        n_samples,
        seed_file=None,
        force=False
    ):

        rets = [ 
                self._check_valid_init(), 
                self._check_valid_backend()
                ] 
        failed = [ ret for ret in rets if ret != self.error_codes.Success ] 
        if len(failed) > 0:
            self.log.warning("Canceling card setup.")            
            return failed

        sample_sizes = self._equal_sample_sizes(
            n_samples,
            sample_limit=100000
        )
        
        if seed_file is not None:
            seeds = np.load(seed_file)
        else:
            seeds = np.random.randint(1, 30081.*30081., len(sample_sizes))

        # check validity of seed input (matching with sample sizes, at the least)
        assert(len(seeds) >= len(sample_sizes))

        self._check_directory(
            local_pathname="cards",
            force=force,
            pattern="card"
        )
        if self.custom_card_directory is not None:
            custom_files = os.listdir(self.custom_card_directory)
        else:
            custom_files = []

        default_files = [f for f in os.listdir(self.module_path + "/data/cards/") if f not in custom_files]
        
        for f in custom_files: 
            shutil.copyfile(
                src=self.custom_card_directory + "/" + f,
                dst=self.dir + "/cards/" + f
            )

        if len(custom_files) > 0: 
            self.log.debug("Copied {} custom card files from directory '{}'".format(len(custom_files), self.custom_card_directory))
        
        for f in default_files:
            shutil.copyfile(
                src=self.module_path + "/data/cards/" + f,
                dst=self.dir + "/cards/" + f
            )

        self.log.debug("Copied {} default card files from directory '{}'".format(len(default_files), self.module_path + "/data/cards/"))

        for i in range(len(sample_sizes)):
            self._replace_lines(
                infile=self.dir + "/cards/run_card.dat", 
                line_numbers=[31,32],
                line_strings=["{} = nevents ! Number of unweighted events requested\n".format(sample_sizes[i]),
                            "{} = iseed   ! (0=assigned automatically=default))\n".format(seeds[i])],
                outfile=self.dir + "/cards/run_card{}.dat".format(i + 1)
            )
        
        self._replace_lines(
            infile=self.dir + "/cards/proc_card.dat",
            line_numbers=[1,2,3],
            line_strings=[
                "import model {}\n".format(self.params['model']),
                "generate {}\n".format(self.params['madgraph_generation_command']),
                ""]
        )

        self.log.debug("Setup {} cards in dir {}".format(
            len(sample_sizes), self.dir + "/cards")
        )

        files = os.listdir(self.dir + "/cards")
        for f in files: 
            self.log.debug(" - \"{}\"".format(f))
        
        return [self.error_codes.Success]

    def run_morphing(
        self,
        morphing_trials=2500,
        force=False
    ):

        rets = [ 
                self._check_valid_backend()
                ] 
        failed = [ ret for ret in rets if ret != self.error_codes.Success ] 
        if len(failed) > 0:
            self.log.warning("Canceling morphing run.")            
            return failed
        
        # check directory for existing morphing information 
        self._check_directory(
                local_pathname="data",
                force=force,
                pattern="madminer_{}.h5".format(self.name)
            )

        # add parameterizations to madminer
        for parameter in self.params['parameters']:
            self.madminer_object.add_parameter(
                    lha_block=self.params['parameters'][parameter]['lha_block'],
                    lha_id=self.params['parameters'][parameter]['lha_id'],
                    parameter_name=parameter,
                    morphing_max_power=self.params['parameters'][parameter]['morphing_max_power'],   
                    parameter_range=self.params['parameters'][parameter]['parameter_range']
                )

        for benchmark in self.params['benchmarks']:
            self.madminer_object.add_benchmark(
                self.params["benchmarks"][benchmark], benchmark
            )

            # for benchmark in self.params['parameters'][parameter]['parameter_benchmarks']:
            #     self.madminer_object.add_benchmark(
            #         {parameter:benchmark[0]},benchmark[1]
            #     )

        self.max_power = max([self.params['parameters'][param]['morphing_max_power'] for param in self.params['parameters']])
        self.madminer_object.set_morphing(
                include_existing_benchmarks=True, 
                max_overall_power=self.max_power,
                n_trials=2500
            )

        # save data
        self.madminer_object.save(self.dir + "/data/madminer_{}.h5".format(self.name))
        self.log.debug("successfully ran morphing.")

        return [self.error_codes.Success]

    def setup_mg5_scripts(
        self,
        samples,
        sample_benchmark,
        force=False,
        mg_dir=None,
        platform="lxplus7",
        use_pythia_card=False
    ):
        
        sample_sizes = self._equal_sample_sizes(samples, sample_limit=100000)

        rets = [ 
                self._check_valid_init(),
                self._check_valid_cards(len(sample_sizes)),
                self._check_valid_morphing(),
                self._check_valid_backend()
                ] 

        failed = [ ret for ret in rets if ret != self.error_codes.Success ] 
        if len(failed) > 0:
            self.log.warning("Canceling mg5 script setup.")            
            return failed

        self._check_directory(
            local_pathname="mg_processes/signal/madminer/scripts",
            force=force,
            mkdir_if_not_existing=False
        )

        self.madminer_object.load(self.dir + "/data/madminer_{}.h5".format(self.name)) 

        # check platform and change initial_command as necessary
        if platform=="lxplus7": 
            initial_command = "source /cvmfs/sft.cern.ch/lcg/views/LCG_94/x86_64-centos7-gcc62-opt/setup.sh; echo \"SOURCED IT\""
            self.log.debug("Ran lxplus7 initial source cmd.")
        elif platform=="pheno": 
            initial_command = 'module purge; module load pheno/pheno-sl7_gcc73; module load cmake/cmake-3.9.6'
        else:
            self.log.error("Platform not recognized. Canceling mg5 script setup.")
            self.log.error("(note: use name 'pheno' for the default belgian server)")
            self.log.error("((I didn't know the proper name, sorry))")
            failed.append(self.error_codes.InvalidPlatformError)

        # init mg_dir
        if mg_dir is not None:
            if not os.path.exists(mg_dir):
                self.log.warning("MGDIR variable '{}' invalid".format(mg_dir))
                self.log.warning("Aborting mg5 script setup routine.")
                failed.append(self.error_codes.NoDirectoryError)
        
        elif(getpass.getuser() == 'pvischia'):
            mg_dir = '/home/ucl/cp3/pvischia/smeft_ml/MG5_aMC_v2_6_2'
        elif(getpass.getuser() == 'alheld'):
            mg_dir = '/home/ucl/cp3/alheld/projects/MG5_aMC_v2_6_2'
        elif(getpass.getuser() == 'llepotti'):
            mg_dir = '/afs/cern.ch/work/l/llepotti/private/MG5_aMC_v2_6_5'
        else:
            self.log.warning("No mg_dir provided and username not recognized. Aborting.")
            failed.append(self.error_codes.NoDirectoryError)

        if len(failed) > 0:
            return failed

        self.log.debug("mg_dir set to '{}'".format(mg_dir))
        # setup pythia card 
        if use_pythia_card:
            pythia_card = self.dir + '/cards/pythia8_card.dat'
        else:
            pythia_card = None

        self.madminer_object.run_multiple(
            sample_benchmarks=[sample_benchmark],
            mg_directory=mg_dir,
            mg_process_directory=self.dir + '/mg_processes/signal',
            proc_card_file=self.dir + '/cards/proc_card.dat',
            param_card_template_file=self.dir + '/cards/param_card_template.dat',
            run_card_files=[self.dir + '/cards/run_card{}.dat'.format(i + 1) for i in range(len(sample_sizes))],
            pythia8_card_file=pythia_card,
            log_directory=self.dir + '/logs/signal',
            initial_command=initial_command,
            only_prepare_script=True
        )

        self._write_config(
            { 'samples' : samples, 'sample_benchmark' : sample_benchmark, 'run_bool' : False },
            self._main_sample_config()
        )
        self.log.debug("Successfully setup mg5 scripts. Ready for execution")
        return [self.error_codes.Success]

    def run_mg5_script(
        self,
        platform,
        samples,
        force=False
    ):
       
        sample_sizes = self._equal_sample_sizes(samples=samples, sample_limit=100000)

        rets = [ 
                self._check_valid_init(), 
                self._check_valid_cards(len(sample_sizes)),
                self._check_valid_morphing(),
                self._check_valid_mg5_scripts(samples),
                self._check_valid_backend()
                ]
        failed = [ ret for ret in rets if ret != self.error_codes.Success ] 

        if len(failed) > 0:
            self.log.warning("Canceling mg5 script run.")            
            return failed

        self._check_directory(
            local_pathname="mg_processes/signal/Events",
            force=force,
            pattern="run_"
        )

        if platform=="lxplus7":
            cmd = "env -i bash -l -c 'source /cvmfs/sft.cern.ch/lcg/views/LCG_94/x86_64-centos7-gcc62-opt/setup.sh; source {}/mg_processes/signal/madminer/run.sh'".format(self.dir)
        elif platform=="pheno":
            self.log.warning("'pheno' platform case selected.")
            self.log.warning("Please note that this platform has not yet been tested with this code.")
            cmd = 'module purge; module load pheno/pheno-sl7_gcc73; module load cmake/cmake-3.9.6'
        else:
            self.log.warning("Platform not recognized. Canceling mg5 script setup.")
            self.log.warning("(note: use name 'pheno' for the default belgian server)")
            self.log.warning("((I didn't know the proper name, sorry))")
            failed.append(self.error_codes.InvalidPlatformError)

        if len(failed) > 0: 
            return failed

        self.log.info("")
        self.log.info("Running mg5 scripts.")
        self.log.info("(This might take awhile - go grab a coffee.)")
        self.log.info("")
        self.log.info("")

        os.system(cmd)

        self.log.info("")
        self.log.info("")
        self.log.info("Finished with data simualtion.")
        self.log.debug("Writing config dictionary and saving simulated parameters.")

        # rewrite run config file with true run_bool
        run_dict = self._load_config(self._main_sample_config())
        run_dict['run_bool'] = True

        self._write_config(
            run_dict,
            self._main_sample_config()
        )

        return [self.error_codes.Success]
    
    def process_mg5_data(
        self
    ):

        rets = [ 
                self._check_valid_mg5_run()
                ]

        failed = [ ret for ret in rets if ret != self.error_codes.Success ] 

        if len(failed) > 0:
            self.log.warning("Canceling mg5 data processing routine.")            
            return failed

        mg5_run_dict = self._load_config(self._main_sample_config())
        samples = mg5_run_dict['samples']
        sample_benchmark = mg5_run_dict['sample_benchmark']

        lhe_processor_object = madminer.lhe.LHEProcessor(filename=self.dir + "/data/madminer_{}.h5".format(self.name))
        n_cards = self._number_of_cards(samples, 100000)
        for i in range(n_cards):
            lhe_processor_object.add_sample(
                self.dir + "/mg_processes/signal/Events/run_{:02d}/unweighted_events.lhe.gz".format(
                    i + 1
                ),
                sampled_from_benchmark=sample_benchmark,
                is_background=False
            )

        for observable in self.params['observables']: 
            lhe_processor_object.add_observable(
                observable,
                self.params['observables'][observable],
                required=True
            )

        lhe_processor_object.analyse_samples()
        lhe_processor_object.save(self.dir + "/data/madminer_{}_with_data_parton.h5".format(self.name))
        return [self.error_codes.Success]

    def plot_mg5_data_corner(
        self,
        image_save_name=None,
        bins=(40,40),
        ranges=[(-8,8),(0,600)]
    ):

        rets = [ 
            self._check_valid_mg5_process()
            ]
        failed = [ ret for ret in rets if ret != self.error_codes.Success ] 

        if len(failed) > 0:
            self.log.warning("Canceling mg5 data plotting.")            
            return failed

        (_, 
        benchmarks, 
        _,_,_,
        observables,
        _,_,_,_) = madminer.utils.interfaces.madminer_hdf5.load_madminer_settings(
            filename = self.dir + "/data/madminer_{}_with_data_parton.h5".format(self.name)
        )

        legend_labels = [label for label in benchmarks]
        labels = [label for label in observables]
        observations = []
        weights = []

        for o, w in madminer.utils.interfaces.madminer_hdf5.madminer_event_loader(
            filename=self.dir + "/data/madminer_{}_with_data_parton.h5".format(self.name)
        ):
            observations.append(o)
            weights.append(w)

        obs = np.squeeze(np.asarray(observations))
        weights = np.squeeze(np.asarray(weights)).T
        norm_weights = np.copy(weights) # normalization factors for plots

        self.log.info("correcting normalizations by total sum of weights per benchmark:")

        for i, weight in enumerate(weights):
            sum_bench = (weight.sum())
            norm_weights[i] /= sum_bench
            self.log.info("{}: {}".format(i + 1, sum_bench))
        
        # labels=[r'$\Delta \eta_{t\bar{t}}$',r'$p_{T, x0}$ [GeV]']

        plt = corner.corner(obs, labels=labels, color='C0', bins=bins, range=ranges, weights=norm_weights[0])
        plt.label = legend_labels[0] 

        for i in range(norm_weights.shape[0] - 1):
            plt_prime = corner.corner(obs, labels=labels, color='C{}'.format(i + 1), bins=bins, range=ranges, weights=norm_weights[i + 1], fig=plt)
            plt_prime.label = legend_labels[i + 1]

    
        full_save_name = "{}/madgraph_data_{}_{}s.png".format(
            self.dir,
            image_save_name,
            obs.shape[0]
        )

        plt.axes[0].autoscale('y')
        plt.axes[3].autoscale('y')
        plt.legend(legend_labels)

        if image_save_name is not None:
            plt.savefig(full_save_name)
        else:
            plt.show()

        return [self.error_codes.Success]

    # training-related member functions

    def train_data(
        self,
        augmented_samples,
        sample_name, 
        training_name,
        augmentation_benchmark,
        n_theta_samples=2500,
        bins=(40,40),
        override_step=None,
        image_save_name=None
    ):

        if override_step is not None: 
            self.TRAINING_STEP = override_step
        else: 
            self.TRAINING_STEP = 0
        
        if self.TRAINING_STEP < 1:
            ret = self.augment_samples(
                    sample_name=sample_name,
                    n_or_frac_augmented_samples=int(augmented_samples),
                    augmentation_benchmark=augmentation_benchmark,
                    n_theta_samples=n_theta_samples
                )
            if self.error_codes.Success not in ret: 
                self.log.warning("Quitting Train Data Function")
                return ret
            self.TRAINING_STEP = 1

        if self.TRAINING_STEP < 2: 
            ret = self.plot_compare_mg5_and_augmented_data(
                    sample_name, 
                    image_save_name=image_save_name,
                    bins=bins,
                    mark_outlier_bins=True
                )
            if self.error_codes.Success not in ret:
                self.log.warning("Quitting Train Data Function")
                return ret
            self.TRAINING_STEP = 2

        return [self.error_codes.Success]

    def augment_samples(
        self,
        sample_name,
        n_or_frac_augmented_samples,
        augmentation_benchmark,
        n_theta_samples=100,
        evaluation_aug_dir=None
    ):
        """
        Augments sample data and saves to a new sample with name <sample_name>.
        This allows for multiple different training sets to be saved on a single sample set.

        parameters:
            sample_name, required:
                string, name of the training data objects to modify
            n_or_frac_augmented_samples, required:
                if type(int), number of samples to draw from simulated madminer data with the sample augmenter
                if type(float), fraction of the simulated samples to draw with the sample augmenter.  
            augmentation_benchmark, required: 
                string, benchmark to feed to trainer

        returns:
            int, error code. 0 is obviously good. 

        """
        # check for processed data
        rets = [ 
            self._check_valid_mg5_process()
            ]
        failed = [ ret for ret in rets if ret != self.error_codes.Success ] 

        if len(failed) > 0:
            self.log.warning("Canceling sample augmentation.")            
            return failed

        # train the ratio
        sample_augmenter = madminer.sampling.SampleAugmenter(
            filename=self.dir + "/data/madminer_{}_with_data_parton.h5".format(self.name)
        )

        if augmentation_benchmark not in sample_augmenter.benchmarks:
            self.log.error("Provided augmentation_benchmark not in given benchmarks!")
            self.log.warning("Please choose from the below existing benchmarks:")
            if sample_augmenter.n_benchmarks < 1:
                self.log.warning(" - None")
            for benchmark in sample_augmenter.benchmarks:
                self.log.info(" - '{}'".format(benchmark))
            failed.append(self.error_codes.InvalidInputError)

        samples = 0

        # if int, this is a direct number
        if type(n_or_frac_augmented_samples) == int: 
            samples = n_or_frac_augmented_samples
        # otherwise, this number represents a fractional quantity
        elif type(n_or_frac_augmented_samples) == float:
            samples = int(n_or_frac_augmented_samples*float(sample_augmenter.n_samples))
        # otherwise we quit
        else:
            self.log.error("Incorrect input ")
            failed.append(self.error_codes.InvalidTypeError )


        if samples > 100000000:
            self.log.warning("Training on {} samples is ridiculous.. reconsider".format(samples))
            self.log.warning("quitting sample augmentation")
            failed.append(self.error_codes.Error)
        if n_theta_samples > int(0.10*samples):
            self.log.warning("Scaling n_theta_samples down for input")
            self.log.warning("Old: {}".format(n_theta_samples)) 
            n_theta_samples = int(0.05*samples)
            self.log.warning("New: {}".format(n_theta_samples))

        if len(failed) > 0:
            return failed

        # parameter ranges
        priors = [('flat',) + self.params['parameters'][parameter]['parameter_range'] for parameter in self.params['parameters']]

        if evaluation_aug_dir is not None:
            aug_dir = evaluation_aug_dir
            config_file = "{}/augmented_sample.mmconfig".format(aug_dir)
        else: 
            aug_dir = self.dir + "/data/samples/{}".format(sample_name)
            config_file = self._augmentation_config(sample_name)

        # train the ratio
        sample_augmenter.extract_samples_train_ratio(
            theta0=madminer.sampling.random_morphing_thetas(n_thetas=n_theta_samples, priors=priors),
            theta1=madminer.sampling.constant_benchmark_theta(augmentation_benchmark),
            n_samples=samples,
            folder=aug_dir,
            filename="augmented_samples_"
        )

        # extract samples at each benchmark
        for benchmark in sample_augmenter.benchmarks:
            sample_augmenter.extract_samples_test(
                theta=madminer.sampling.constant_benchmark_theta(benchmark),
                n_samples=samples,
                folder=aug_dir,
                filename='augmented_samples_{}'.format(benchmark)
            )

        # save augmentation config file
        self._write_config(
            {
                "augmentation_benchmark": augmentation_benchmark,
                "augmentation_samples": samples,
                "theta_samples": n_theta_samples,
                "sample_name": sample_name
            },
            config_file
        )

        return [self.error_codes.Success]

    def plot_augmented_data_corner(
        self,
        sample_name,
        image_save_name=None,
        bins=(40,40),
        ranges=[(-8,8),(0,600)],
        max_index=0
    ):
        rets = [ 
            self._check_valid_augmented_data(sample_name=sample_name),
            self._check_valid_mg5_process()
            ]
        failed = [ ret for ret in rets if ret != self.error_codes.Success ] 

        if len(failed) > 0:
            self.log.warning("Canceling augmented sampling plots.")            
            return failed

        search_key = "x_augmented_samples_"

        x_files = [f for f in os.listdir(self.dir + "/data/samples/{}".format(sample_name)) if search_key in f]
        
        x_arrays = dict([(f[len(search_key):][:-len(".npy")], np.load(self.dir + "/data/samples/{}/".format(sample_name) + f)) for f in x_files])
        
        x_size = max([x_arrays[obs].shape[0] for obs in x_arrays])

        # grab benchmarks and observables from files
        (_, 
        benchmarks, 
        _,_,_,
        observables,
        _,_,_,_) = madminer.utils.interfaces.madminer_hdf5.load_madminer_settings(
            filename = self.dir + "/data/madminer_{}_with_data_parton.h5".format(self.name)
        )

        legend_labels = [label for label in benchmarks]
        labels = [label for label in observables]
        # alternate labels?? here they be
        # labels=[r'$\Delta \eta_{t\bar{t}}$',r'$p_{T, x0}$ [GeV]']

        plt = corner.corner(x_arrays[legend_labels[0]], labels=labels, color='C0', bins=bins, range=ranges)
        plt.label = legend_labels[0] 

        for i,benchmark in enumerate(legend_labels[1:]):
            plt_prime = corner.corner(x_arrays[benchmark], labels=labels, color='C{}'.format(i + 1), bins=bins, range=ranges, fig=plt)
            plt_prime.label = legend_labels[i + 1]

        full_save_name = "{}/data/samples/{}/augmented_data_{}_{}s.png".format(
            self.dir,
            sample_name,
            image_save_name,
            x_size
        )

        plt.axes[0].autoscale('y')
        plt.axes[3].autoscale('y')
        plt.legend(legend_labels)

        if image_save_name is not None:
            plt.savefig(full_save_name)
        else: 
            plt.show()

        return [self.error_codes.Success]

    def plot_compare_mg5_and_augmented_data(
        self,
        sample_name,
        image_save_name=None,
        mark_outlier_bins=False,
        bins=(40,40),
        ranges=[(-8,8),(0,600)],
        dens=True,
        alphas=(0.8, 0.4),
        figlen=5, 
        threshold=2.0
    ):

        err, x_aug, x_mg5 = self._get_mg5_and_augmented_arrays(
                sample_name, 
                bins, 
                ranges, 
                dens
            )

        if self.error_codes.Success not in err:
            self.log.warning("Quitting mg5 vs augmented data plot comparison")
            return err

        (_,
        benchmarks,
        _,_,_,
        observables
        ,_,_,_,_) = madminer.utils.interfaces.madminer_hdf5.load_madminer_settings(
            filename = self.dir + "/data/madminer_{}_with_data_parton.h5".format(self.name)
        )

        # create lists of each variable
        benchmark_list = [benchmark for benchmark in benchmarks]

        y_fac = 1.0 #np.diff(x_mg5[1][:,:])

        mg5_x = x_mg5[1][:,:,:-1]
        mg5_y = x_mg5[0][:,:]*y_fac
 
        mg5_y_err = x_mg5[3][:,:]*y_fac
        mg5_y_err_x = 0.5*(x_mg5[1][:,:,1:] + x_mg5[1][:,:,:-1])

        aug_x = x_aug[1][:,:,:-1]
        aug_y = x_aug[0][:,:]*y_fac

        flag_x = (x_aug[1][:,:,:-1] + np.diff(x_aug[1][:,:])/2.0)

        r, pers = self._compare_mg5_and_augmented_data(
                x_aug, 
                x_mg5,
                y_fac,
                threshold
            )

        fig, axs = plt.subplots(1, x_aug[0].shape[0], figsize=(figlen*x_aug[0].shape[0], figlen))
        for i in range(x_aug[0].shape[0]):
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
            height_step = np.max([mg5_y[i], aug_y[i]])/40.0
            # counts = np.zeros(mg5_x[i,0].shape)
            for j in range(x_aug[0].shape[1]):

                # plot augmented and mg5 histograms
                axs[i].plot(mg5_x[i,j], mg5_y[i,j], colors[j], label="{} mg5".format(benchmark_list[j]), drawstyle="steps-post", alpha=alphas[0])
                axs[i].plot(aug_x[i,j], aug_y[i,j], colors[j], label="{} aug".format(benchmark_list[j]), drawstyle="steps-post", alpha=alphas[1])
                
                # plot errorbars
                axs[i].errorbar(mg5_y_err_x[i,j], mg5_y[i,j], yerr=mg5_y_err[i,j], fmt='none', capsize=1.5, elinewidth=1., ecolor='black', alpha=alphas[1])
                
                # if needed, mark outlier bins with a character
                if mark_outlier_bins:
                    index = r[i,j] >= threshold
                    axs[i].plot(flag_x[i,j][index],
                       -height_step*(float(j) + 1.0)*np.ones(flag_x[i,j][index].shape),
                        linestyle="None", marker="x", color=colors[j])
    
        for i,observable in enumerate(observables): 
            axs[i].set_xlabel(observable) 
            axs[i].set_yticklabels([])

        handles = []
        labels = []
        
        for ax in axs: 
            handles += (ax.get_legend_handles_labels()[0])
            labels += (ax.get_legend_handles_labels()[1])
        
        by_label = collections.OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        fig.tight_layout()

        self.log.info("MG5 Samples: {}".format(x_mg5[2]))
        self.log.info("Aug Samples: {}".format(x_aug[2]))

        self._tabulate_comparison_information(r, pers, observables, benchmarks, threshold)

        if image_save_name is not None:
            full_save_name = "{}/mg5_vs_augmented_data_{}_{}s.png".format(
                self.dir,
                image_save_name,
                x_aug[0].shape[0]
            )
            plt.savefig(full_save_name)
            plt.clf()
        else:
            plt.show()

        return [self.error_codes.Success] 

    def train_method(
        self, 
        sample_name,
        training_name,
        training_method="alices",
        node_architecture=(100,100,100),
        n_epochs=30,
        batch_size=128,
        activation_function='relu',
        trainer='adam',
        initial_learning_rate=0.001,
        final_learning_rate=0.0001
    ):

        known_training_methods = ["alices", "alice"]

        rets = [ 
            self._check_valid_augmented_data(sample_name=sample_name),
            ]
        failed = [ ret for ret in rets if ret != self.error_codes.Success ] 

        if len(failed) > 0:
            self.log.warning("Quitting train_method function.")            
            return failed

        if training_method not in known_training_methods:
            self.log.error("Unknown raining method {}".format(training_method))
            self.log.info("Try again with one of:")
            for method in known_training_methods:
                self.log.info(" - {}".format(method))
            self.log.warning("Quitting train_method function.")
            return self.error_codes.UnknownTrainingModelError

        existing_files = glob.glob("{}/models/{}/{}_{}*".format(self.dir, sample_name, training_name, training_method)) 
        if len(existing_files) > 0:
            self.log.warning("There are trained models with this name!")
            for fname in existing_files: 
                self.log.warning(" - {}".format(fname))
            self.log.warning("Rerun function with a different name, or delete previously trained models.")
            return self.error_codes.ExistingModelError

        # load madminer H5 file??
        # self.madminer_object.load()   
        
        forge = madminer.ml.MLForge()
        forge.train(
                method=training_method,
                theta0_filename='{}/data/samples/{}/theta0_train.npy'.format(self.dir, sample_name),
                x_filename='{}/data/samples/{}/x_train.npy'.format(self.dir, sample_name),
                y_filename='{}/data/samples/{}/y_train.npy'.format(self.dir, sample_name),
                r_xz_filename='{}/data/samples/{}/r_xz_train.npy'.format(self.dir, sample_name),
                t_xz0_filename='{}/data/samples/{}/t_xz_train.npy'.format(self.dir, sample_name),
                n_hidden=node_architecture,
                activation=activation_function,
                n_epochs=n_epochs,
                batch_size=batch_size,
                trainer=trainer,
                initial_lr=initial_learning_rate,
                final_lr=final_learning_rate
            )
            
        # size = self._dir_size(
        #     pathname="{}/models/{}".format(self.dir, sample_name),
        #     matching_pattern=["{}".format(training_name), "{}_settings.json".format(training_method)]
        # )
 
        # if size > 0:
        #     training_name = "{}{}".format(training_name, size)

        forge.save('{}/models/{}/{}/train'.format(self.dir, sample_name, training_name))

        self._write_config(
            {
                'training_method' : training_method, 
                'training_name' : training_name,
                'node_architecture' : node_architecture,
                'n_epochs' : n_epochs,
                'batch_size' : batch_size,
                'activation_function' : activation_function,
                'trainer' : trainer,
                'initial_learning_rate' : initial_learning_rate,
                'final_learning_rate' : final_learning_rate,
                'sample_name' : sample_name
            },
            self._training_config(sample_name, training_name)
        )

        return self.error_codes.Success

    def evaluate_method(
        self,
        training_name, 
        evaluation_name,
        evaluation_samples,
        theta_grid_spacing=40,
        evaluation_benchmark=None,
        sample_name="*"
    ):
        params = locals() 
        for parameter in params:
            if parameter is not 'self': 
                self.log.debug("{}: {}".format(parameter, params[parameter]))
        # self.log.debug("training name: {}".format(training_name))
        # self.log.debug("evaluation name: {}".format(evaluation_name))
        # self.log.debug("evaluation samples: {}".format(evaluation_samples))
        # self.log.debug("sample name: {}".format(sample_name))

        rets = [ 
            self._check_valid_trained_models(training_name=training_name, sample_name=sample_name),
            ]

        failed = [ret for ret in rets if ret != self.error_codes.Success]

        if len(failed) > 0:
            self.log.warning("Quitting train_method function.")            
            return failed
        
        fname = glob.glob("{}/models/{}/{}/train_settings.json".format(self.dir, sample_name, training_name))[0]
        
        model_params = self._load_config(
            "{}/training_model.mmconfig".format(os.path.dirname(fname))
            )
        sample_params = self._load_config(
            self._augmentation_config(model_params['sample_name'])
            )

        for path_to_check in [ 
            "{}/evaluations/".format(self.dir),
            "{}/evaluations/{}/".format(self.dir, model_params['training_name']),
            ]:
            if not os.path.exists(path_to_check):
                os.mkdir(path_to_check)
                
        evaluation_dir = "{}/evaluations/{}/{}/".format(self.dir, model_params['training_name'], evaluation_name)
        
        self._write_config(
            {
                'evaluation_name': evaluation_name,
                'training_name': training_name,
                'evaluation_samples': evaluation_samples,
                'evaluation_benchmark': evaluation_benchmark
            },
            self._evaluation_config(training_name, evaluation_name)
        )
        return  

        if os.path.exists(evaluation_dir):
            if len([f for f in os.listdir(evaluation_dir) if "log_r_hat" in f]) > 0:
                self.log.error("Identically sampled, trained, and named evaluation instance already exists!! Pick another.")
                self.log.error(" - {}".format(evaluation_dir))
                return [self.error_codes.ExistingEvaluationError] 
        else:
            os.mkdir(evaluation_dir)

        self.log.info("evaluating trained method '{}'".format(model_params['training_name']))
        self.log.debug("Model Params: ")
        for spec in model_params:
            self.log.debug(" - {}: {}".format(spec, model_params[spec]))

        self.log.debug("")
        self.log.debug("Aug. Sample Params: ")
        for spec in sample_params:
            self.log.debug(" - {}: {}".format(spec, sample_params[spec]))

        forge = madminer.ml.MLForge()
        sample_augmenter = madminer.sampling.SampleAugmenter(
                filename=self.dir + "/data/madminer_{}_with_data_parton.h5".format(self.name)
            )
        forge.load("{}/train".format(os.path.dirname(self._training_config(model_params['sample_name'], model_params['training_name']))))

        theta_grid = np.mgrid[[slice(*tup, theta_grid_spacing*1.0j) for tup in [self.params['parameters'][parameter]['parameter_range'] for parameter in self.params['parameters']]]].T
        theta_dim = theta_grid.shape[-1]

        if evaluation_benchmark is None: 
            evaluation_benchmark = sample_params['augmentation_benchmark']

        evaluation_sample_config = self._check_for_matching_augmented_data(evaluation_samples, evaluation_benchmark)
        
        if evaluation_sample_config is None: 
            self.augment_samples(
                sample_name="{}_eval_augment".format(evaluation_name),
                n_or_frac_augmented_samples=evaluation_samples,
                augmentation_benchmark=evaluation_benchmark,
                n_theta_samples=sample_params['theta_samples'],
                evaluation_aug_dir=evaluation_dir
            )
            evaluation_sample_config = '{}/augmented_sample.mmconfig'.format(evaluation_dir)

        # stack parameter grid into (N**M X M) size vector (tragic scale factor :--< ) 
        for i in range(theta_dim): 
            theta_grid = np.vstack(theta_grid)
        
        np.save("{}/theta_grid.npy".format(evaluation_dir), theta_grid)
        
        log_r_hat_dict = {}

        for benchmark in sample_augmenter.benchmarks: 
            ret = forge.evaluate(
                theta0_filename="{}/theta_grid.npy".format(evaluation_dir),
                x='{}/x_augmented_samples_{}.npy'.format(evaluation_dir, benchmark)
            )
            log_r_hat_dict[benchmark] = ret[0]

        for benchmark in log_r_hat_dict: 
            np.save("{}/log_r_hat_{}.npy".format(evaluation_dir, benchmark), log_r_hat_dict[benchmark])
            self.log.info("log_r_hat info saved for benchmark {}: 'log_r_hat_{}.npy'".format(benchmark, benchmark))

        self._write_config(
            {
                'evaluation_name': evaluation_name,
                'training_name': training_name,
                'evaluation_samples': evaluation_samples,
                'evaluation_benchmark': evaluation_benchmark
            },
            self._evaluation_config(training_name, evaluation_name)
        )

        return self.error_codes.Success

    def plot_evaluation_results(
        self,
        evaluation_name,
        training_name
    ):
        self.log.info("Plotting evaluation results for evaluation instance '{}'".format(evaluation_name))

        evaluations = self.list_evaluations()
        evaluation_tuples = [evaluation for evaluation in evaluations if evaluation[1]['evaluation_name'] == evaluation_name] 

        if len(evaluation_tuples) == 0:
            self.log.error("Evaluation name '{}' not found")
            self.log.error("Please choose oneof the following evaluations:")
            for evaluation in self.list_evaluations():
                self.log.error(" - {}".format(evaluation[1]['evaluation_name']))
            return self.error_codes.NoEvaluatedModelError
        elif len(evaluation_tuples) > 1:
            self.log.error("Mutiple matching evaluations found. Please specify")

        theta = np.load( "/theta_grid*"))

        return self.error_codes.Success
