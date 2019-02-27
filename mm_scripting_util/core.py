
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
        
        ret = self._search_for_paths(custom_card_directory, include_module_paths=False)

        if ret is None: 
            self.log.error("Selected custom card directory '{}' could not be found.".format(custom_card_directory))
            self.log.error("Using default card directory instead.")
            self.custom_card_directory = None
        else:
            self.log.debug("Using custom card directory '{}'".format(ret))
            self.custom_card_directory = ret

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
        if not self._check_valid_init():
            return
        self._remove_files(
                self.dir,
                include_folder=True
            )

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
                self.setup_cards(
                        n_samples=samples,
                        seed_file=seed_file,
                        force=force
                    )
                self.STEP = 1

            if self.STEP < 2:
                self.run_morphing(
                        force=force,
                        morphing_trials=morphing_trials
                    )
                self.STEP = 2

            if self.STEP < 3:        
                self.setup_mg5_scripts(
                        samples=samples,
                        sample_benchmark=sample_benchmark,
                        force=force,
                        mg_dir=mg_dir,
                        platform=platform,
                        use_pythia_card=use_pythia_card,
                    )
                self.STEP = 3

            if self.STEP < 4:        
                self.run_mg5_script(
                        platform=platform,
                        samples=samples,
                        force=force
                    )
                self.STEP = 4

            if self.STEP < 5:
                self.process_mg5_data(
                        samples=samples, 
                        sample_benchmark=sample_benchmark
                    )
                self.STEP = 5

        except:
            self.log.error(traceback.format_exc())
            self.log.error("ABORTING")

        return 0

    def setup_cards(
            self, 
            n_samples,
            seed_file=None,
            force=False
        ):
        
        if not self._check_valid_init():
            os.mkdir(self.dir)

        if not self._check_valid_backend():
            self.log.warning("Canceling card setup.")            
            return 1

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
        
        return 0

    def run_morphing(
            self,
            morphing_trials=2500,
            force=False
        ):

        if not self._check_valid_backend():
            self.log.warning("Canceling morphing run.")            
            return 1

        
        # check directory for existing morphing information 
        self._check_directory(
                local_pathname="data",
                force=force,
                pattern="madminer_example.h5"
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
        self.madminer_object.save(self.dir + '/data/madminer_example.h5')
        self.log.debug("successfully ran morphing.")

        return 0

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

        # validate and check that cards, morphing, and backend are correctly installed 
        if not (self._check_valid_init() and 
                self._check_valid_cards(len(sample_sizes)) and
                self._check_valid_morphing() and 
                self._check_valid_backend()):
            self.log.warning("Canceling mg5 script setup.")
            return 1

        self._check_directory(
            local_pathname="mg_processes/signal/madminer/scripts",
            force=force,
            mkdir_if_not_existing=False
        )

        self.madminer_object.load(self.dir + '/data/madminer_example.h5') 

        # check platform and change initial_command as necessary
        if platform=="lxplus7": 
            initial_command = "source /cvmfs/sft.cern.ch/lcg/views/LCG_94/x86_64-centos7-gcc62-opt/setup.sh "
        elif platform=="pheno": 
            initial_command = 'module purge; module load pheno/pheno-sl7_gcc73; module load cmake/cmake-3.9.6'
        else:
            self.log.warning("Platform not recognized. Canceling mg5 script setup.")
            self.log.warning("(note: use name 'pheno' for the default belgian server)")
            self.log.warning("((I didn't know the proper name, sorry))")
            return 1

        # init mg_dir
        if mg_dir is not None:
            if not os.path.exists(mg_dir):
                self.log.warning("MGDIR variable '{}' invalid".format(mg_dir))
                self.log.warning("Aborting mg5 script setup routine.")
                return 1
        
        elif(getpass.getuser() == 'pvischia'):
            mg_dir = '/home/ucl/cp3/pvischia/smeft_ml/MG5_aMC_v2_6_2'
        elif(getpass.getuser() == 'alheld'):
            mg_dir = '/home/ucl/cp3/alheld/projects/MG5_aMC_v2_6_2'
        elif(getpass.getuser() == 'llepotti'):
            mg_dir = '/afs/cern.ch/work/l/llepotti/private/MG5_aMC_v2_6_5'
        else:
            self.log.warning("No mg_dir provided and username not recognized. Aborting.")
            return 1
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
        return 0

    def run_mg5_script(
            self,
            platform,
            samples,
            force=False
        ):
       
        sample_sizes = self._equal_sample_sizes(samples=samples, sample_limit=100000)

        if not (self._check_valid_init() and 
                self._check_valid_cards(len(sample_sizes)) and
                self._check_valid_morphing() and 
                self._check_valid_mg5_scripts(samples) and
                self._check_valid_backend()):
            self.log.warning("Canceling mg5 run.")
            return 1

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
            return 1

        self.log.info("")
        self.log.info("Running mg5 scripts.")
        self.log.info("(This might take awhile - go grab a coffee.)")
        self.log.info("")
        self.log.info("")

        os.system(cmd)
    
    def process_mg5_data(
            self,
            samples, 
            sample_benchmark
        ):

        if not self._check_valid_mg5_run(samples):
            self.log.warning("Quitting mg5 data processing.")
            return 1
        
        lhe_processor_object = madminer.lhe.LHEProcessor(filename=self.dir + "/data/madminer_example.h5")
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
        lhe_processor_object.save(self.dir + "/data/madminer_example_with_data_parton.h5")

    def plot_mg5_data(
            self,
            image_save_name=None
        ):

        if not self._check_valid_mg5_process():
            self.log.warning("Quitting mg5 data plotting")
            return 1

        observations = None
        weights = None

        with h5py.File(self.dir + "/data/madminer_example_with_data_parton.h5", "r") as f:
            observations = f["samples/observations"]
            weights = f["samples/weights"]

        if None in [observations, weights]:
            self.log.warning("Observations or weights of sample is None")
            self.log.debug("observations: ")
            self.log.debug(observations)
            self.log.debug("weights: ")
            self.log.debut(weights)
            return 1


        obs = np.asarray([observations[obs] for obs in observations]).T
        weights = np.asarray([weights[weight] for weight in weights])
        norm_weights = np.copy(weights) # normalization factors for plots

        print("correcting normalizations by total sum of weights per benchmark:")
        for i, weight in enumerate(weights):
            sum_bench = (weights[weight].sum())
            norm_weights[i] /= sum_bench
            print(sum_bench)
                    
        labels=[r'$\Delta \eta_{t\bar{t}}$',r'$p_{T, x0}$ [GeV]']
        bins=(30,30)
        ranges = [(-8,8), (0,600)]


        fig = corner.corner(obs, labels=labels, color='C1',
                            bins=bins, range=ranges,
                            weights=norm_weights[1])
        fig2 = corner.corner(obs, labels=labels, color='C2',
                             bins=bins, range=ranges,
                             weights=norm_weights[2], fig=fig)
        fig3 = corner.corner(obs, labels=labels, color='C0',
                             bins=bins, range=ranges, 
                             weights=norm_weights[0], fig=fig)

        full_save_name = "{}/madgraph_data_{}_{}s.png".format(
            self.dir,
            image_save_name,
            obs.shape[0]
        )

        if full_save_name is not None:
            plt.savefig(full_save_name)
        else:
            plt.show()


        # blue: SM, orange: CP-odd, green: mixture
        return 0

    # training-related member functions