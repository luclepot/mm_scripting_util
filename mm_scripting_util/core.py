
from .util import *


"""
Main driver container for the class. 

This class should handle all necessary interfacing with
madminer, in the context of the ttH CP process. 

"""

class miner(mm_util):   
    
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
            morphing_trials=2500
        ):
        """
        Standard data simulation run. Should go from start to finish with data simulation.
        """
        try:
            self.STEP = self._get_simulation_step(
                self._number_of_cards(samples, 100000),
                samples
            )

            if self.STEP < 1:
                ret = self.setup_cards(
                        n_samples=samples,
                        seed_file=seed_file,
                        force=force
                    )
                if len(ret) > 0:
                    return ret
                self.STEP = 1

            if self.STEP < 2:
                ret = self.run_morphing(
                        force=force,
                        morphing_trials=morphing_trials
                    )
                if len(ret) > 0:
                    return ret
                self.STEP = 2

            if self.STEP < 3:        
                ret = self.setup_mg5_scripts(
                        samples=samples,
                        sample_benchmark=sample_benchmark,
                        force=force,
                        mg_dir=mg_dir,
                        platform=platform,
                        use_pythia_card=use_pythia_card,
                    )
                if len(ret) > 0:
                    return ret
                self.STEP = 3

            if self.STEP < 4:        
                ret = self.run_mg5_script(
                        platform=platform,
                        samples=samples,
                        force=force
                    )
                if len(ret) > 0:
                    return ret
                self.STEP = 4

            if self.STEP < 5:
                ret = self.process_mg5_data(
                        samples=samples, 
                        sample_benchmark=sample_benchmark
                    )
                if len(ret) > 0:
                    return ret
                self.STEP = 5

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
            initial_command = "source /cvmfs/sft.cern.ch/lcg/views/LCG_94/x86_64-centos7-gcc62-opt/setup.sh "
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

        return [self.error_codes.Success]
    
    def process_mg5_data(
            self,
            samples, 
            sample_benchmark
        ):

        rets = [ 
                self._check_valid_mg5_run(samples)
                ]
        failed = [ ret for ret in rets if ret != self.error_codes.Success ] 

        if len(failed) > 0:
            self.log.warning("Canceling mg5 data processing routine.")            
            return failed

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

    def plot_mg5_data(
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
        plt.show()

        return [self.error_codes.Success]

    # training-related member functions

    def train_data(
            self,
            augmented_samples, 
            training_name,
            augmentation_benchmark,
            n_theta_samples=2500
        ):

        self.augment_samples(
                training_name=training_name,
                n_or_frac_augmented_samples=int(augmented_samples),
                augmentation_benchmark=augmentation_benchmark,
                n_theta_samples=n_theta_samples
            )

        return [self.error_codes.Success]

    def augment_samples(
            self,
            training_name,
            n_or_frac_augmented_samples,
            augmentation_benchmark,
            n_theta_samples=2500
        ):
        """
        Augments sample data and saves to a new sample with name <training_name>.
        This allows for multiple different training sets to be saved on a single sample set.

        parameters:
            training_name, required:
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

        if samples > 10000000:
            self.log.warning("Training on {} samples is ridiculous.. reconsider".format(samples))
            self.log.warning("quitting sample augmentation")
            failed.append(self.error_codes.Error)

        if len(failed) > 0:
            return failed

        # parameter ranges
        priors = [('flat',) + self.params['parameters'][parameter]['parameter_range'] for parameter in self.params['parameters']]

        # train the ratio
        sample_augmenter.extract_samples_train_ratio(
            theta0=madminer.sampling.random_morphing_thetas(n_thetas=n_theta_samples, priors=priors),
            theta1=madminer.sampling.constant_benchmark_theta(augmentation_benchmark),
            n_samples=samples,
            folder=self.dir + "/data/samples",
            filename=training_name + "_train"
        )

        for benchmark in sample_augmenter.benchmarks:
            sample_augmenter.extract_samples_test(
                theta=madminer.sampling.constant_benchmark_theta(benchmark),
                n_samples=samples,
                folder=self.dir + "/data/samples",
                filename='{}_augmented_samples_{}'.format(training_name, benchmark)                 
            )

        return [self.error_codes.Success]

    def plot_augmented_data(
            self,
            training_name,
            image_save_name=None,
            bins=(40,40),
            ranges=[(-8,8),(0,600)],
            max_index=0
        ):
        rets = [ 
            self._check_valid_augmented_data(training_name=training_name),
            self._check_valid_mg5_process()
            ]
        failed = [ ret for ret in rets if ret != self.error_codes.Success ] 

        if len(failed) > 0:
            self.log.warning("Canceling augmented sampling plots.")            
            return failed

        search_key = "x_{}_augmented_samples_".format(training_name)

        x_files = [f for f in os.listdir(self.dir + "/data/samples") if search_key in f]
        
        x_arrays = dict([(f[len(search_key):][:-len(".npy")], np.load(self.dir + "/data/samples/" + f)) for f in x_files])
        
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

        full_save_name = "{}/augmented_data_{}_{}s.png".format(
            self.dir,
            image_save_name,
            x_size
        )

        plt.axes[0].autoscale('y')
        plt.axes[3].autoscale('y')
        plt.legend(legend_labels)

        if image_save_name is not None:
            plt.savefig(full_save_name)
        plt.show()

        return [self.error_codes.Success]

    def get_histogram_error(
            self,
            training_name,
            display_plots=False,
            image_save_name=None,
            bins=(40,40),
            ranges=[(-8,8),(0,600)],
            dens=False
        ):

        # normal error checking
        rets = [ 
            self._check_valid_augmented_data(training_name=training_name),
            self._check_valid_mg5_process()
            ]
        failed = [ ret for ret in rets if ret != self.error_codes.Success ] 

        if len(failed) > 0:
            self.log.warning("Canceling augmented sampling plots.")            
            return failed

        # search key for augmented samples
        search_key = "x_{}_augmented_samples_".format(training_name)
        x_files = [f for f in os.listdir(self.dir + "/data/samples") if search_key in f]        
        x_arrays = dict([(f[len(search_key):][:-len(".npy")], np.load(self.dir + "/data/samples/" + f)) for f in x_files])
        x_size = max([x_arrays[obs].shape[0] for obs in x_arrays])

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

        mg5_observations = []
        mg5_weights = []

        for o, w in madminer.utils.interfaces.madminer_hdf5.madminer_event_loader(
            filename=self.dir + "/data/madminer_{}_with_data_parton.h5".format(self.name)
        ):
            mg5_observations.append(o)
            mg5_weights.append(w)

        mg5_obs = np.squeeze(np.asarray(mg5_observations))
        mg5_weights = np.squeeze(np.asarray(mg5_weights)).T
        mg5_norm_weights = np.copy(mg5_weights) # normalization factors for plots

        self.log.info("correcting normalizations by total sum of weights per benchmark:")

        for i, weight in enumerate(mg5_weights):
            sum_bench = (weight.sum())
            mg5_norm_weights[i] /= sum_bench
            # self.log.info("{}: {}".format(i + 1, sum_bench))

        x_list_augmented = np.asarray([
                [np.asarray(np.histogram(
                    x_arrays[benchmark][:,i],
                    bins=bins[i],
                    range=ranges[i],
                    weights=(mg5_obs[:,i].size/x_arrays[benchmark][:,i].size)*np.ones(x_arrays[benchmark][:,0].shape)*mg5_norm_weights[0][0],
                    density=dens
                )) for benchmark in benchmark_list]
                for i in range(len(observable_list)) 
            ])

        x_list_mg5 = np.asarray([
                [np.histogram(
                    mg5_obs[:,i], 
                    range=ranges[i],
                    bins=bins[i],
                    weights=weight,
                    density=dens
                ) for weight in mg5_norm_weights] 
                for i in range(len(mg5_obs[0]))
            ])

        figlen = 5

        if display_plots:
            fig, axs = plt.subplots(1, x_list_augmented.shape[0], figsize=(figlen*x_list_augmented.shape[0], figlen))
            for i in range(x_list_augmented.shape[0]):
                colors = ['b', 'r', 'g']
                for j in range(x_list_augmented.shape[1]):
                    axs[i].step(x_list_augmented[i,j,1][:-1], x_list_augmented[i,j,0], colors[j], label="augmented_{}".format(j + 1))
                    axs[i].step(x_list_mg5[i,j,1][:-1], x_list_mg5[i,j,0], colors[j], label="mg5_{}".format(i + 1))
                axs[i].legend()
            # fig.set_figheight = figlen
            # fig.set_figwidth = figlen*3. # *x_list_augmented.shape[0]
            fig.tight_layout()
            plt.show()

        return mg5_norm_weights, x_arrays, mg5_obs
        # return [self.error_codes.Success], x_list_augmented, x_list_mg5, mg5_norm_weights
