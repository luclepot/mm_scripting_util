from mm_scripting_util.util import * 
from mm_scripting_util.util import _mm_util, _mm_backend_util, _mm_base_util

class miner(_mm_util):
    """
    Main container for the class. 

    This class should handle all necessary interfacing with
    madminer and madgraph. 

    """

    __MODULE_NAME = "mm_scripting_util"

    # general class member functions

    def __init__(
        self,
        name,
        backend,
        card_directory=None,
        path=None,
        loglevel=logging.INFO,
        madminer_loglevel=logging.INFO,
        init_loglevel=logging.INFO,
        autodestruct=False,
        _cmd_line_origin=False,
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
            card_directory:
                string, path to a card directory from which to load template cards, if one desires to switch the current template cards out for new ones.
        """

        self._cmd_line_origin = _cmd_line_origin

        if path is None:
            path = os.getcwd()

        # initialize helper classes

        _mm_base_util.__init__(self, name, path)

        _mm_backend_util.__init__(self)

        self.autodestruct = autodestruct
        self.log = logging.getLogger(__name__)
        self.module_path = os.path.dirname(__file__)

        self.set_loglevel(loglevel)
        self.set_loglevel(madminer_loglevel, module="madminer")
        
        self.name = name
        self.dir = "{}/{}".format(self.path, self.name)

        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
            self.log.log(init_loglevel, "Creating new directory " + self.dir)
        else:
            self.log.log(init_loglevel, "Initialized to existing directory " + self.dir)

        self.log.log(init_loglevel, 
            "Initialized a new miner object with name '{}'".format(self.name)
        )
        self.log.log(init_loglevel, "- module path at {}".format(self.module_path))
        self.log.log(init_loglevel, "- new miner object path at " + self.dir)

        self.madminer_object = madminer.core.MadMiner()
        self.lhe_processor_object = None

        self.log.log(init_loglevel, "Loading custom card directory... ")
        self.card_directory = None

        # backend param should refer to the name, not a specific backend filepath
        self.backend = backend.replace('.dat', '')
            

        # if card_directory is specified.. 
        if card_directory is not None:
            for path_check in [
                card_directory,
                "{}/data/{}".format(self.module_path, card_directory)
            ]:
                if os.path.exists(path_check):
                    self.card_directory = path_check
            if self.card_directory is None:
                self.log.error(
                    "Selected card directory '{}' could not be found.".format(
                        card_directory
                    )
                )
                self.log.error("Using default card directory instead.")
                self.card_directory = self.default_card_directory    
        # else, check using the backend parameter
        else:
            for path_check in [
                "cards_{}".format(self.backend),
                "{}/data/cards_{}".format(self.module_path, self.backend)
            ]:
                if os.path.exists(path_check):
                    self.card_directory = path_check
            if self.card_directory is None:
                self.log.error("No card directory found using auto-spec backend {}".format(self.backend))
                self.log.error("Using default card directory instead.")
                self.card_directory = self.default_card_directory

        self._load_backend("{}.dat".format(self.backend))

        self.log.log(init_loglevel, "Using card directory '{}',".format(self.card_directory))
        self.log.log(init_loglevel, "with {} files".format(len(os.listdir(self.card_directory))))

    def set_loglevel(
        self,
        loglevel,
        module=None
    ):

        logging.basicConfig(
            format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s",
            datefmt="%H:%M",
            level=logging.WARNING,
        )

        if module is None:
            module = self.__MODULE_NAME

        logging.getLogger(module).setLevel(loglevel)

        return loglevel

    def destroy_sample(
        self
    ):
        rets = [self._check_valid_init()]
        failed = [ret for ret in rets if ret != self.error_codes.Success]

        if len(failed) > 0:
            return failed

        self._remove_files(self.dir, include_folder=True)

        return [self.error_codes.Success]

    def list_samples(
        self, 
        verbose=False, 
        criteria='*', 
        include_info=False,
    ):

        sample_list = glob.glob(
            "{}/data/samples/{}/augmented_sample.mmconfig".format(self.dir, criteria)
        ) + glob.glob("{}/evaluations/*/{}/augmented_sample.mmconfig".format(self.dir, criteria))

        return self._list_verbose_helper('augmented samples', sample_list, verbose, criteria, 'sample_name', include_info)
        
    def list_models(
        self, 
        verbose=False, 
        criteria='*', 
        include_info=False,
    ):
        
        model_list = glob.glob("{}/models/*/{}/training_model.mmconfig".format(self.dir, criteria))

        return self._list_verbose_helper('trained models', model_list, verbose, criteria, 'training_name', include_info)

    def list_evaluations(
        self, 
        verbose=False, 
        criteria='*', 
        include_info=False,
    ):

        evaluation_list = glob.glob(
            "{}/evaluations/*/{}/evaluation.mmconfig".format(self.dir, criteria)
        )
        return self._list_verbose_helper('evaluations', evaluation_list, verbose, criteria, 'evaluation_name', include_info)

    @staticmethod
    def list_backends():
        return os.listdir("{}/data/backends/".format(os.path.dirname(__file__)))

    @staticmethod
    def list_cards():
        return glob.glob("{}/data/*cards*".format(os.path.dirname(__file__)))
    
    @staticmethod
    def list_full_backends():
        backends = [w.replace('.dat', '') for w in miner.list_backends()]
        cards = [card.split('/')[-1].replace('cards_', '') for card in miner.list_cards()]
        return set(backends).intersection(cards)

    @staticmethod
    def list_existing_samples():
        possible_folders = [f for f in os.listdir() if os.path.isdir(f) and f[0] != '.']
    # simulation-related member functions

    def simulate_data(
        self,
        samples,
        sample_benchmark,
        seed_file=None,
        force=True,
        mg_dir=None,
        use_pythia_card=False,
        mg_environment_cmd='ubc',
        mg_run_cmd='ubc',
        morphing_trials=2500,
        override_step=None,
    ):
        """
        Standard data simulation run. Should go from start to finish with data simulation.
        """
        try:
            if override_step is not None:
                self.SIMULATION_STEP = override_step
            else:
                self.SIMULATION_STEP = self._get_simulation_step(
                    samples
                )

            if self.SIMULATION_STEP < 1 or force:
                self.log.debug("")
                self.log.debug("RUNNING SETUP CARDS, STEP 1")
                self.log.debug("")
                ret = self.setup_cards(
                    n_samples=samples, seed_file=seed_file, force=force
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
                ret = self.run_morphing(force=force, morphing_trials=morphing_trials)
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
                    mg_environment_cmd=mg_environment_cmd,
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
                    mg_environment_cmd=mg_run_cmd, samples=samples, force=force
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
        force=False,
        run_card_modifications=[
            " = nevents ! Number of unweighted events requested",
            " = iseed   ! (0=assigned automatically=default))",
        ],
    ):

        rets = [self._check_valid_init(), self._check_valid_backend()]
        failed = [ret for ret in rets if ret != self.error_codes.Success]
        if len(failed) > 0:
            self.log.warning("Canceling card setup.")
            return failed

        sample_sizes = self._equal_sample_sizes(n_samples, sample_limit=100000)

        if seed_file is not None:
            seeds = np.load(seed_file)
        else:
            seeds = np.random.randint(1, 30081.0 * 30081.0, len(sample_sizes))

        # check validity of seed input (matching with sample sizes, at the least)
        assert len(seeds) >= len(sample_sizes)

        self._check_directory(local_pathname="cards", force=force, pattern="card")

        files = os.listdir(self.card_directory)
        filenames = {}

        for f in files:
            shutil.copyfile(
                src=self.card_directory + "/" + f, dst=self.dir + "/cards/" + f
            )
            filenames[f] = "{}/cards/{}".format(self.card_directory, f)

        self.log.info(
            "Copied {} card files from directory '{}'".format(
                len(files), self.card_directory
            )
        )

        #
        # SETUP RUN CARDS
        #
        run_card_filenames = {f: filenames[f] for f in filenames if "run_card" in f}

        for i in range(len(sample_sizes)):
            for f in run_card_filenames:
                nums, names = self._get_keyword_filenumbers(
                    run_card_modifications, f, fdir=self.card_directory
                )
                values = [sample_sizes[i], seeds[i]]
                self._replace_lines(
                    infile="{}/cards/{}".format(self.dir, f),
                    line_numbers=nums,
                    line_strings=[
                        "{}{}\n".format(values[j], name) for j, name in enumerate(names)
                    ],
                    outfile=self.dir
                    + "/cards/{}{}.dat".format(f.replace(".dat", ""), i + 1),
                )

        self.log.debug(
            "Setup {} cards in dir {}".format(len(sample_sizes), self.dir + "/cards")
        )

        files = os.listdir(self.dir + "/cards")
        for f in files:
            self.log.debug(' - "{}"'.format(f))

        #
        # SETUP PROC CARDS // disabled for now, just do it manually (easy enough man)
        #
        # proc_card_filenames = { f : filenames[f] for f in filenames if "proc_card" in f }
        # possible_proc_card_changes = [
        #     ("madgraph_generation_command", "generate "),
        #     ("model", "import model ")
        # ]

        # for f in proc_card_filenames:
        #     for key in self.params:
        #         for change, change_syntax in possible_proc_card_changes:
        #             if change in key:
        #                 self._replace_lines(
        #                     infile="{}/cards/{}".format(self.dir, f),
        #                     line_numbers=self._get_keyword_filenumbers([change_syntax], f, fdir=self.card_directory)[0],
        #                     line_strings=["{}{}\n".format(change_syntax, change)],
        #                     outfile="{}/cards/{}.dat".format(self.dir, f.replace('.dat',''))
        #                 )

        return [self.error_codes.Success]

    def run_morphing(
        self,
        morphing_trials=2500,
        force=False
    ):
        rets = [self._check_valid_backend()]
        failed = [ret for ret in rets if ret != self.error_codes.Success]
        if len(failed) > 0:
            self.log.warning("Canceling morphing run.")
            return failed

        # check directory for existing morphing information
        self._check_directory(
            local_pathname="data",
            force=force,
            pattern="madminer_{}.h5".format(self.name),
        )

        # add parameterizations to madminer
        for parameter in self.params["parameters"]:
            self.madminer_object.add_parameter(
                lha_block=self.params["parameters"][parameter]["lha_block"],
                lha_id=self.params["parameters"][parameter]["lha_id"],
                parameter_name=parameter,
                morphing_max_power=self.params["parameters"][parameter][
                    "morphing_max_power"
                ],
                parameter_range=self.params["parameters"][parameter]["parameter_range"],
            )

        for benchmark in self.params["benchmarks"]:
            self.madminer_object.add_benchmark(
                self.params["benchmarks"][benchmark], benchmark
            )

            # for benchmark in self.params['parameters'][parameter]['parameter_benchmarks']:
            #     self.madminer_object.add_benchmark(
            #         {parameter:benchmark[0]},benchmark[1]
            #     )

        self.max_power = max(
            [
                self.params["parameters"][param]["morphing_max_power"]
                for param in self.params["parameters"]
            ]
        )
        self.madminer_object.set_morphing(
            include_existing_benchmarks=True,
            max_overall_power=self.max_power,
            n_trials=2500,
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
        mg_environment_cmd='lxplus7',
        use_pythia_card=False,
    ):

        sample_sizes = self._equal_sample_sizes(samples, sample_limit=100000)

        rets = [
            self._check_valid_init(),
            self._check_valid_cards(),
            self._check_valid_morphing(),
            self._check_valid_backend(),
        ]

        failed = [ret for ret in rets if ret != self.error_codes.Success]
        if len(failed) > 0:
            self.log.warning("Canceling mg5 script setup.")
            return failed

        self._check_directory(
            local_pathname="mg_processes/signal/madminer/scripts",
            force=force,
            mkdir_if_not_existing=False,
        )

        self.madminer_object.load(self.dir + "/data/madminer_{}.h5".format(self.name))

        # check platform and change initial_command as necessary
        if mg_environment_cmd == "lxplus7":
            initial_command = 'source /cvmfs/sft.cern.ch/lcg/views/LCG_94/x86_64-centos7-gcc62-opt/setup.sh; echo "SOURCED IT"'
            self.log.debug("Ran lxplus7 initial source cmd.")
        elif mg_environment_cmd == "pheno":
            initial_command = "module purge; module load pheno/pheno-sl7_gcc73; module load cmake/cmake-3.9.6"
        elif mg_environment_cmd == "ubc":
            initial_command = "exec env -i bash -l -c 'which python; printenv'"
        else:
            initial_command = mg_environment_cmd
        
        self.log.debug('mg env command: {}'.format(initial_command))

        # init mg_dir
        if mg_dir is not None:
            if not os.path.exists(mg_dir):
                self.log.warning("MGDIR variable '{}' invalid".format(mg_dir))
                self.log.warning("Aborting mg5 script setup routine.")
                failed.append(self.error_codes.NoDirectoryError)
        elif len(glob.glob("../MG5_aMC_*")) > 0:
            mg_dir = os.path.abspath(glob.glob("../MG5_aMC_*")[0])
        elif getpass.getuser() == "pvischia":
            mg_dir = "/home/ucl/cp3/pvischia/smeft_ml/MG5_aMC_v2_6_2"
        elif getpass.getuser() == "alheld":
            mg_dir = "/home/ucl/cp3/alheld/projects/MG5_aMC_v2_6_2"
        elif getpass.getuser() == "llepotti":
            mg_dir = "/afs/cern.ch/work/l/llepotti/private/MG5_aMC_v2_6_5"
        else:
            self.log.warning(
                "No mg_dir provided and username not recognized. Aborting."
            )
            failed.append(self.error_codes.NoDirectoryError)

        if len(failed) > 0:
            return failed

        self.log.debug("mg_dir set to '{}'".format(mg_dir))
        # setup pythia card
        if use_pythia_card:
            pythia_card = self.dir + "/cards/pythia8_card.dat"
        else:
            pythia_card = None

        for param in [pythia_card, mg_dir, initial_command,]:
            self.log.debug(" - {}".format(param))

        self.madminer_object.run_multiple(
            sample_benchmarks=[sample_benchmark],
            mg_directory=mg_dir,
            mg_process_directory=self.dir + "/mg_processes/signal",
            proc_card_file=self.dir + "/cards/proc_card.dat",
            param_card_template_file=self.dir + "/cards/param_card_template.dat",
            run_card_files=[
                self.dir + "/cards/run_card{}.dat".format(i + 1)
                for i in range(len(sample_sizes))
            ],
            pythia8_card_file=pythia_card,
            log_directory=self.dir + "/logs/signal",
            initial_command=initial_command,
            only_prepare_script=True,
        )

        self._write_config(
            {
                "samples": samples,
                "sample_benchmark": sample_benchmark,
                "run_bool": False,
            },
            self._main_sample_config(),
        )
        self.log.debug("Successfully setup mg5 scripts. Ready for execution")
        return [self.error_codes.Success]

    def run_mg5_script(
        self,
        mg_environment_cmd,
        samples,
        force=False
    ):

        rets = [
            self._check_valid_init(),
            self._check_valid_cards(),
            self._check_valid_morphing(),
            self._check_valid_mg5_scripts(samples),
            self._check_valid_backend(),
        ]
        failed = [ret for ret in rets if ret != self.error_codes.Success]

        if len(failed) > 0:
            self.log.warning("Canceling mg5 script run.")
            return failed

        self._check_directory(
            local_pathname="mg_processes/signal/Events", force=force, pattern="run_"
        )

        if mg_environment_cmd == "lxplus7":
            cmd = "env -i bash -l -c 'source /cvmfs/sft.cern.ch/lcg/views/LCG_94/x86_64-centos7-gcc62-opt/setup.sh; source {}/mg_processes/signal/madminer/run.sh'".format(self.dir)
        elif mg_environment_cmd == "pheno":
            self.log.warning("'pheno' platform case selected.")
            self.log.warning(
                "Please note that this platform has not yet been tested with this code."
            )
            cmd = "module purge; module load pheno/pheno-sl7_gcc73; module load cmake/cmake-3.9.6"
        elif mg_environment_cmd == 'ubc':
            cmd = "exec env -i bash -l -c 'which python; source {}/mg_processes/signal/madminer/run.sh'".format(self.dir)
        else:
            cmd = mg_environment_cmd

        # if cmd.strip()[-1] == "'":
        #     cmd = cmd.strip()[0:-1]

       
        self.log.debug('mg env command: {}'.format(cmd))
        # self.log.warning("Platform not recognized. Canceling mg5 script setup.")
        # self.log.warning("(note: use name 'pheno' for the default belgian server)")
        # self.log.warning("((I didn't know the proper name, sorry))")
        # failed.append(self.error_codes.InvalidPlatformError)

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
        run_dict["run_bool"] = True

        self._write_config(run_dict, self._main_sample_config())

        return [self.error_codes.Success]

    def process_mg5_data(
        self
    ):

        rets = [self._check_valid_mg5_run()]

        failed = [ret for ret in rets if ret != self.error_codes.Success]

        if len(failed) > 0:
            self.log.warning("Canceling mg5 data processing routine.")
            return failed

        mg5_run_dict = self._load_config(self._main_sample_config())
        samples = mg5_run_dict["samples"]
        sample_benchmark = mg5_run_dict["sample_benchmark"]

        lhe_processor_object = madminer.lhe.LHEProcessor(
            filename=self.dir + "/data/madminer_{}.h5".format(self.name)
        )
        n_cards = self._number_of_cards(samples, 100000)
        for i in range(n_cards):
            lhe_processor_object.add_sample(
                self.dir
                + "/mg_processes/signal/Events/run_{:02d}/unweighted_events.lhe.gz".format(
                    i + 1
                ),
                sampled_from_benchmark=sample_benchmark,
                is_background=False,
            )

        for observable in self.params["observables"]:
            lhe_processor_object.add_observable(
                observable, self.params["observables"][observable], required=True
            )

        lhe_processor_object.analyse_samples()
        lhe_processor_object.save(
            self.dir + "/data/madminer_{}_with_data_parton.h5".format(self.name)
        )
        return [self.error_codes.Success]

    def plot_mg5_data_corner(
        self,
        image_save_name=None,
        bins=40,
        ranges=None,
        include_automatic_benchmarks=True,
    ):

        rets = [self._check_valid_mg5_process()]
        failed = [ret for ret in rets if ret != self.error_codes.Success]

        if len(failed) > 0:
            self.log.warning("Canceling mg5 data plotting.")
            return failed

        (
            _,
            benchmarks,
            _,
            _,
            _,
            observables,
            _,
            _,
            _,
            _,
        ) = madminer.utils.interfaces.madminer_hdf5.load_madminer_settings(
            filename=self.dir
            + "/data/madminer_{}_with_data_parton.h5".format(self.name)
        )

        include_array = None

        if not include_automatic_benchmarks:
            include_array = [
                i for i, bm in enumerate(benchmarks) if bm in self.params["benchmarks"]
            ]
            benchmarks = {
                bm: benchmarks[bm]
                for bm in benchmarks
                if bm in self.params["benchmarks"]
            }

        legend_labels = [label for label in benchmarks]
        labels = [label for label in observables]

        if type(bins != list):
            bins = [bins for i in range(len(observables))]

        obs, _, norm_weights, _ = self._get_raw_mg5_arrays(include_array=include_array)

        self.log.info(
            "correcting normalizations by total sum of weights per benchmark:"
        )

        if ranges is None:
            ranges = self._get_automatic_ranges(obs, norm_weights)
            self.log.info("No ranges specified, using automatic range finding.")

        assert len(bins) == len(observables) == len(ranges)

        # labels=[r'$\Delta \eta_{t\bar{t}}$',r'$p_{T, x0}$ [GeV]']

        plt = corner.corner(
            obs,
            labels=labels,
            color="C0",
            bins=bins,
            range=ranges,
            weights=norm_weights[0],
        )
        plt.label = legend_labels[0]

        for i in range(norm_weights.shape[0] - 1):
            plt_prime = corner.corner(
                obs,
                labels=labels,
                color="C{}".format(i + 1),
                bins=bins,
                range=ranges,
                weights=norm_weights[i + 1],
                fig=plt,
            )
            plt_prime.label = legend_labels[i + 1]

        full_save_name = "{}/data/madgraph_data_{}.png".format(
            self.dir, image_save_name
        )

        plt.axes[0].autoscale("y")
        plt.axes[3].autoscale("y")
        plt.legend(legend_labels)

        full_save_name = "{}/data/madgraph_data_{}.png".format(
            self.dir, image_save_name if image_save_name is not None else 'temp'
        )   
    
        if self._cmd_line_origin:
            self.log.debug('showing graph via feh... (cmd line interface triggered)')
            plt.savefig(full_save_name)
            subprocess.Popen(['feh', full_save_name ])
        elif image_save_name is not None:
            self.log.debug('saving image to \'{}\''.format(full_save_name))
            plt.savefig(full_save_name)
        else:
            self.log.debug('displaying image...')
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
        bins=None,
        ranges=None,
        override_step=None,
        image_save_name=None,
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
                n_theta_samples=n_theta_samples,
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
                ranges=ranges,
                mark_outlier_bins=True,
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
        evaluation_aug_dir=None,
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
        rets = [self._check_valid_mg5_process()]
        failed = [ret for ret in rets if ret != self.error_codes.Success]

        if len(failed) > 0:
            self.log.warning("Canceling sample augmentation.")
            return failed

        # train the ratio
        sample_augmenter = madminer.sampling.SampleAugmenter(
            filename=self.dir
            + "/data/madminer_{}_with_data_parton.h5".format(self.name)
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
            samples = int(
                n_or_frac_augmented_samples * float(sample_augmenter.n_samples)
            )
        # otherwise we quit
        else:
            self.log.error("Incorrect input ")
            failed.append(self.error_codes.InvalidTypeError)

        if samples > 100000000:
            self.log.warning(
                "Training on {} samples is ridiculous.. reconsider".format(samples)
            )
            self.log.warning("quitting sample augmentation")
            failed.append(self.error_codes.Error)
        if n_theta_samples > int(0.10 * samples):
            self.log.warning("Scaling n_theta_samples down for input")
            self.log.warning("Old: {}".format(n_theta_samples))
            n_theta_samples = int(0.05 * samples)
            self.log.warning("New: {}".format(n_theta_samples))

        if len(failed) > 0:
            return failed

        # parameter ranges
        priors = [
            ("flat",) + self.params["parameters"][parameter]["parameter_range"]
            for parameter in self.params["parameters"]
        ]
        self.log.debug("Priors: ")

        for prior in priors:
            self.log.debug(" - {}".format(prior))

        if evaluation_aug_dir is not None:
            aug_dir = evaluation_aug_dir
            config_file = "{}/augmented_sample.mmconfig".format(aug_dir)
        else:
            aug_dir = self.dir + "/data/samples/{}".format(sample_name)
            config_file = self._augmentation_config(sample_name)

        # train the ratio

        sample_augmenter.extract_samples_train_ratio(
            theta0=madminer.sampling.random_morphing_thetas(
                n_thetas=n_theta_samples, priors=priors
            ),
            theta1=madminer.sampling.constant_benchmark_theta(augmentation_benchmark),
            n_samples=samples,
            folder=aug_dir,
            filename="augmented_sample_ratio",
        )

        # extract samples at each benchmark
        for benchmark in sample_augmenter.benchmarks:
            sample_augmenter.extract_samples_test(
                theta=madminer.sampling.constant_benchmark_theta(benchmark),
                n_samples=samples,
                folder=aug_dir,
                filename="augmented_samples_{}".format(benchmark),
            )

        # save augmentation config file
        self._write_config(
            {
                "augmentation_benchmark": augmentation_benchmark,
                "augmentation_samples": samples,
                "theta_samples": n_theta_samples,
                "sample_name": sample_name,
                "all_benchmarks": dict(sample_augmenter.benchmarks),
            },
            config_file,
        )

        return [self.error_codes.Success]

    def plot_augmented_data_corner(
        self,
        sample_name,
        image_save_name=None,
        bins=None,
        ranges=None,
        include_automatic_benchmarks=True,
    ):
        rets = [
            self._check_valid_augmented_data(sample_name=sample_name),
            self._check_valid_mg5_process(),
        ]
        failed = [ret for ret in rets if ret != self.error_codes.Success]

        if len(failed) > 0:
            self.log.warning("Canceling augmented sampling plots.")
            return failed

        search_key = "x_augmented_samples_"

        x_files = [
            f
            for f in os.listdir(self.dir + "/data/samples/{}".format(sample_name))
            if search_key in f
        ]

        x_arrays = dict(
            [
                (
                    f[len(search_key) :][: -len(".npy")],
                    np.load(self.dir + "/data/samples/{}/".format(sample_name) + f),
                )
                for f in x_files
            ]
        )

        x_size = max([x_arrays[obs].shape[0] for obs in x_arrays])

        # grab benchmarks and observables from files
        (
            _,
            benchmarks,
            _,
            _,
            _,
            observables,
            _,
            _,
            _,
            _,
        ) = madminer.utils.interfaces.madminer_hdf5.load_madminer_settings(
            filename=self.dir
            + "/data/madminer_{}_with_data_parton.h5".format(self.name)
        )

        if not include_automatic_benchmarks:
            benchmarks = {
                bm: benchmarks[bm]
                for bm in benchmarks
                if bm in self.params["benchmarks"]
            }
            x_arrays = {
                bm: x_arrays[bm] for bm in x_arrays if bm in self.params["benchmarks"]
            }

        legend_labels = [label for label in benchmarks]
        labels = [label for label in observables]

        default_bins = 40

        if bins is None:
            bins = default_bins 
        if not hasattr(bins, '__iter__'):
            bins  = [bins for i in range(len(labels))]

        if ranges is None:
            ranges = np.mean(
                [
                    [np.min(x_arrays[bm], axis=0), np.max(x_arrays[bm], axis=0)]
                    for bm in x_arrays
                ],
                axis=0,
            ).T

        # alternate labels?? here they be
        # labels=[r'$\Delta \eta_{t\bar{t}}$',r'$p_{T, x0}$ [GeV]']
        assert len(labels) == len(bins) == len(ranges)

        plt = corner.corner(
            x_arrays[legend_labels[0]],
            labels=labels,
            color="C0",
            bins=bins,
            range=ranges,
        )

        plt.label = legend_labels[0]

        for i, benchmark in enumerate(legend_labels[1:]):
            plt_prime = corner.corner(
                x_arrays[benchmark],
                labels=labels,
                color="C{}".format(i + 1),
                bins=bins,
                range=ranges,
                fig=plt,
            )
            plt_prime.label = legend_labels[i + 1]


        plt.axes[0].autoscale("y")
        plt.axes[3].autoscale("y")
        plt.legend(legend_labels)


        full_save_name = "{}/data/samples/{}/augmented_data_{}.png".format(
            self.dir, sample_name, image_save_name if image_save_name is not None else 'temp'
        )   
    
        if self._cmd_line_origin:
            plt.savefig(full_save_name)
            subprocess.Popen(['feh', full_save_name ])
        elif image_save_name is not None:
            plt.savefig(full_save_name)
        else:
            plt.show()
        
        return [self.error_codes.Success]

    def plot_compare_mg5_and_augmented_data(
        self,
        sample_name,
        image_save_name=None,
        mark_outlier_bins=False,
        bins=40,
        ranges=None,
        dens=True,
        alphas=(0.8, 0.4),
        figlen=5,
        threshold=2.0,
        include_automatic_benchmarks=True,
    ):

        err, x_aug, x_mg5 = self._get_mg5_and_augmented_arrays(
            sample_name,
            bins,
            ranges,
            dens,
            include_automatic_benchmarks=include_automatic_benchmarks,
            params=self.params,
        )

        if self.error_codes.Success not in err:
            self.log.warning("Quitting mg5 vs augmented data plot comparison")
            return err

        (
            _,
            benchmarks,
            _,
            _,
            _,
            observables,
            _,
            _,
            _,
            _,
        ) = madminer.utils.interfaces.madminer_hdf5.load_madminer_settings(
            filename=self.dir
            + "/data/madminer_{}_with_data_parton.h5".format(self.name)
        )

        if not include_automatic_benchmarks:
            benchmarks = {
                bm: benchmarks[bm]
                for bm in benchmarks
                if bm in self.params["benchmarks"]
            }

        # create lists of each variable
        benchmark_list = [benchmark for benchmark in benchmarks]

        y_fac = 1.0  # np.diff(x_mg5[1][:,:])

        mg5_x = x_mg5[1][:, :, :-1]
        mg5_y = x_mg5[0][:, :] * y_fac

        mg5_y_err = x_mg5[3][:, :] * y_fac
        mg5_y_err_x = 0.5 * (x_mg5[1][:, :, 1:] + x_mg5[1][:, :, :-1])

        aug_x = x_aug[1][:, :, :-1]
        aug_y = x_aug[0][:, :] * y_fac

        flag_x = x_aug[1][:, :, :-1] + np.diff(x_aug[1][:, :]) / 2.0

        r, pers = self._compare_mg5_and_augmented_data(x_aug, x_mg5, y_fac, threshold)

        fig, axs = plt.subplots(
            1, x_aug[0].shape[0], figsize=(figlen * x_aug[0].shape[0], figlen)
        )

        for i in range(x_aug[0].shape[0]):
            colors = [
                "tab:blue",
                "tab:orange",
                "tab:green",
                "tab:red",
                "tab:purple",
                "tab:brown",
                "tab:pink",
                "tab:gray",
                "tab:olive",
                "tab:cyan",
            ]
            height_step = np.max([mg5_y[i], aug_y[i]]) / 40.0
            # counts = np.zeros(mg5_x[i,0].shape)
            for j in range(x_aug[0].shape[1]):

                # plot augmented and mg5 histograms
                axs[i].plot(
                    mg5_x[i, j],
                    mg5_y[i, j],
                    colors[j],
                    label="{} mg5".format(benchmark_list[j]),
                    drawstyle="steps-post",
                    alpha=alphas[0],
                )
                axs[i].plot(
                    aug_x[i, j],
                    aug_y[i, j],
                    colors[j],
                    label="{} aug".format(benchmark_list[j]),
                    drawstyle="steps-post",
                    alpha=alphas[1],
                )

                # plot errorbars
                axs[i].errorbar(
                    mg5_y_err_x[i, j],
                    mg5_y[i, j],
                    yerr=mg5_y_err[i, j],
                    fmt="none",
                    capsize=1.5,
                    elinewidth=1.0,
                    ecolor="black",
                    alpha=alphas[1],
                )

                # if needed, mark outlier bins with a character
                if mark_outlier_bins:
                    index = r[i, j] >= threshold
                    axs[i].plot(
                        flag_x[i, j][index],
                        -height_step
                        * (float(j) + 1.0)
                        * np.ones(flag_x[i, j][index].shape),
                        linestyle="None",
                        marker="x",
                        color=colors[j],
                    )

        for i, observable in enumerate(observables):
            axs[i].set_xlabel(observable)
            axs[i].set_yticklabels([])

        handles = []
        labels = []

        for ax in axs:
            handles += ax.get_legend_handles_labels()[0]
            labels += ax.get_legend_handles_labels()[1]

        by_label = collections.OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        fig.tight_layout()

        self.log.info("MG5 Samples: {}".format(x_mg5[2]))
        self.log.info("Aug Samples: {}".format(x_aug[2]))

        self._tabulate_comparison_information(
            r, pers, observables, benchmarks, threshold
        )
        
        full_save_name = "{}/data/samples/{}/mg5_vs_augmented_data_{}s.png".format(
            self.dir, sample_name, image_save_name if image_save_name is not None else 'temp'
        )
    
        if self._cmd_line_origin:
            plt.savefig(full_save_name)
            subprocess.Popen(['feh', full_save_name ])
        elif image_save_name is not None:
            plt.savefig(full_save_name)
        else:
            plt.show()
        
        return [self.error_codes.Success]

    def train_method(
        self,
        sample_name,
        training_name,
        training_method="alices",
        node_architecture=(100, 100, 100),
        n_epochs=30,
        batch_size=128,
        activation_function="relu",
        trainer="adam",
        initial_learning_rate=0.001,
        final_learning_rate=0.0001,
    ):

        known_training_methods = ["alices", "alice"]

        rets = [self._check_valid_augmented_data(sample_name=sample_name), self._check_valid_madminer_ml()]
        failed = [ret for ret in rets if ret != self.error_codes.Success]

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

        existing_files = glob.glob(
            "{}/models/{}/{}_{}*".format(
                self.dir, sample_name, training_name, training_method
            )
        )
        if len(existing_files) > 0:
            self.log.warning("There are trained models with this name!")
            for fname in existing_files:
                self.log.warning(" - {}".format(fname))
            self.log.warning(
                "Rerun function with a different name, or delete previously trained models."
            )
            return self.error_codes.ExistingModelError

        # load madminer H5 file??
        # self.madminer_object.load()

        forge = madminer.ml.MLForge()
        forge.train(
            method=training_method,
            theta0_filename="{}/data/samples/{}/theta0_augmented_sample_ratio.npy".format(
                self.dir, sample_name
            ),
            x_filename="{}/data/samples/{}/x_augmented_sample_ratio.npy".format(
                self.dir, sample_name
            ),
            y_filename="{}/data/samples/{}/y_augmented_sample_ratio.npy".format(
                self.dir, sample_name
            ),
            r_xz_filename="{}/data/samples/{}/r_xz_augmented_sample_ratio.npy".format(
                self.dir, sample_name
            ),
            t_xz0_filename="{}/data/samples/{}/t_xz_augmented_sample_ratio.npy".format(
                self.dir, sample_name
            ),
            n_hidden=node_architecture,
            activation=activation_function,
            n_epochs=n_epochs,
            batch_size=batch_size,
            optimizer=trainer,
            initial_lr=initial_learning_rate,
            final_lr=final_learning_rate,
        )

        # size = self._dir_size(
        #     pathname="{}/models/{}".format(self.dir, sample_name),
        #     matching_pattern=["{}".format(training_name), "{}_settings.json".format(training_method)]
        # )

        # if size > 0:
        #     training_name = "{}{}".format(training_name, size)

        forge.save("{}/models/{}/{}/train".format(self.dir, sample_name, training_name))

        self._write_config(
            {
                "training_method": training_method,
                "training_name": training_name,
                "node_architecture": node_architecture,
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "activation_function": activation_function,
                "trainer": trainer,
                "initial_learning_rate": initial_learning_rate,
                "final_learning_rate": final_learning_rate,
                "sample_name": sample_name,
            },
            self._training_config(sample_name, training_name),
        )

        return self.error_codes.Success

    def evaluate_method(
        self,
        training_name,
        evaluation_name,
        evaluation_samples,
        theta_grid_spacing=40,
        evaluation_benchmark=None,
        sample_name="*",
    ):
        params = locals()
        for parameter in params:
            if parameter is not "self":
                self.log.debug("{}: {}".format(parameter, params[parameter]))
        # self.log.debug("training name: {}".format(training_name))
        # self.log.debug("evaluation name: {}".format(evaluation_name))
        # self.log.debug("evaluation samples: {}".format(evaluation_samples))
        # self.log.debug("sample name: {}".format(sample_name))

        rets = [
            self._check_valid_trained_models(
                training_name=training_name, sample_name=sample_name
            ),
            self._check_valid_madminer_ml()
        ]

        failed = [ret for ret in rets if ret != self.error_codes.Success]

        

        if len(failed) > 0:
            self.log.warning("Quitting train_method function.")
            return failed

        fname = glob.glob(
            "{}/models/{}/{}/train_settings.json".format(
                self.dir, sample_name, training_name
            )
        )[0]

        model_params = self._load_config(
            "{}/training_model.mmconfig".format(os.path.dirname(fname))
        )
        sample_params = self._load_config(
            self._augmentation_config(model_params["sample_name"])
        )

        for path_to_check in [
            "{}/evaluations/".format(self.dir),
            "{}/evaluations/{}/".format(self.dir, model_params["training_name"]),
        ]:
            if not os.path.exists(path_to_check):
                os.mkdir(path_to_check)

        evaluation_dir = "{}/evaluations/{}/{}/".format(
            self.dir, model_params["training_name"], evaluation_name
        )

        if os.path.exists(evaluation_dir):
            if len([f for f in os.listdir(evaluation_dir) if "log_r_hat" in f]) > 0:
                self.log.error(
                    "Identically sampled, trained, and named evaluation instance already exists!! Pick another."
                )
                self.log.error(" - {}".format(evaluation_dir))
                return [self.error_codes.ExistingEvaluationError]
        else:
            os.mkdir(evaluation_dir)

        self.log.info(
            "evaluating trained method '{}'".format(model_params["training_name"])
        )
        self.log.debug("Model Params: ")
        for spec in model_params:
            self.log.debug(" - {}: {}".format(spec, model_params[spec]))

        self.log.debug("")
        self.log.debug("Aug. Sample Params: ")
        for spec in sample_params:
            self.log.debug(" - {}: {}".format(spec, sample_params[spec]))

        forge = madminer.ml.MLForge()
        sample_augmenter = madminer.sampling.SampleAugmenter(
            filename=self.dir
            + "/data/madminer_{}_with_data_parton.h5".format(self.name)
        )
        forge.load(
            "{}/train".format(
                os.path.dirname(
                    self._training_config(
                        model_params["sample_name"], model_params["training_name"]
                    )
                )
            )
        )

        theta_grid = np.mgrid[
            [
                slice(*tup, theta_grid_spacing * 1.0j)
                for tup in [
                    self.params["parameters"][parameter]["parameter_range"]
                    for parameter in self.params["parameters"]
                ]
            ]
        ].T
        theta_dim = theta_grid.shape[-1]

        if evaluation_benchmark is None:
            evaluation_benchmark = sample_params["augmentation_benchmark"]

        evaluation_sample_config = self._check_for_matching_augmented_data(
            evaluation_samples, evaluation_benchmark
        )

        if evaluation_sample_config is None:
            self.augment_samples(
                sample_name="{}_eval_augment".format(evaluation_name),
                n_or_frac_augmented_samples=evaluation_samples,
                augmentation_benchmark=evaluation_benchmark,
                n_theta_samples=sample_params["theta_samples"],
                evaluation_aug_dir=evaluation_dir,
            )
            evaluation_sample_config = "{}/augmented_sample.mmconfig".format(
                evaluation_dir
            )

        # stack parameter grid into (N**M X M) size vector (tragic scale factor :--< )
        for i in range(theta_dim):
            theta_grid = np.vstack(theta_grid)

        np.save("{}/theta_grid.npy".format(evaluation_dir), theta_grid)

        log_r_hat_dict = {}

        for benchmark in sample_augmenter.benchmarks:
            ret = forge.evaluate(
                theta0_filename="{}/theta_grid.npy".format(evaluation_dir),
                x="{}/x_augmented_samples_{}.npy".format(
                    os.path.dirname(evaluation_sample_config), benchmark
                ),
            )
            log_r_hat_dict[benchmark] = ret[0]

        for benchmark in log_r_hat_dict:
            np.save(
                "{}/log_r_hat_{}.npy".format(evaluation_dir, benchmark),
                log_r_hat_dict[benchmark],
            )
            self.log.info(
                "log_r_hat info saved for benchmark {}: 'log_r_hat_{}.npy'".format(
                    benchmark, benchmark
                )
            )

        self._write_config(
            {
                "evaluation_name": evaluation_name,
                "training_name": training_name,
                "evaluation_samples": evaluation_samples,
                "evaluation_benchmark": evaluation_benchmark,
                "evaluation_datasets": {
                    key: "{}/{}/log_r_hat_{}.npy".format(self.name, evaluation_dir.split("{}/".format(self.name))[-1], key)
                    for key in log_r_hat_dict
                },
            },
            self._evaluation_config(training_name, evaluation_name),
        )

        return self.error_codes.Success

    def plot_evaluation_results(
        self,
        evaluation_name,
        training_name=None,
        z_contours=[1.0],
        fill_contours=True,
        bb_b=1.16,
        bb_m=0.05,
    ):
        self.log.info(
            "Plotting evaluation results for evaluation instance '{}'".format(
                evaluation_name
            )
        )

        evaluations = self.list_evaluations()
        evaluation_tuples = [
            evaluation
            for evaluation in evaluations
            if evaluation[1]["evaluation_name"] == evaluation_name
        ]

        if training_name is not None:
            evaluation_tuples = list(
                filter(
                    lambda elt: elt[1]["training_name"] == training_name,
                    evaluation_tuples,
                )
            )

        if len(evaluation_tuples) == 0:
            self.log.error("Evaluation name '{}' not found")
            self.log.error("Please choose oneof the following evaluations:")
            for evaluation in self.list_evaluations():
                self.log.error(" - {}".format(evaluation[1]["evaluation_name"]))
            return self.error_codes.NoEvaluatedModelError
        elif len(evaluation_tuples) > 1:
            self.log.error("Mutiple matching evaluations found. Please specify")
            for evaluation_tuple in evaluation_tuples:
                self.log.error(" - {}".format(evaluation_tuple[1]["evaluation_name"]))
                self.log.error("   AT PATH: {}".format(evaluation_tuple[0]))
                self.log.error(
                    "   WITH TRAINING PARENT {}".format(
                        evaluation_tuple[1]["training_name"]
                    )
                )
            return self.error_codes.MultipleMatchingFilesError

        # else tuple is CLEAN, with len 1
        evaluation_tuple = evaluation_tuples[0]
        evaluation_dir = os.path.dirname(evaluation_tuple[0])
        self.log.debug(evaluation_tuple)
        theta_grid = np.load("{}/theta_grid.npy".format(evaluation_dir))
        log_r_hat_dict = {
            key: np.load(evaluation_tuple[1]["evaluation_datasets"][key].replace('//', '/'))
            for key in evaluation_tuple[1]["evaluation_datasets"]
        }

        if len(z_contours) > 0:
            alphas = self._scale_to_range_flipped([0.0] + z_contours, [0.05, 0.5])[
                1:
            ]
        else:
            alphas = []

        for p_num, parameter in enumerate(self.params["parameters"]):
            for i, benchmark in enumerate(log_r_hat_dict):
                mu = np.mean(log_r_hat_dict[benchmark], axis=1)
                sigma = np.std(log_r_hat_dict[benchmark], axis=1)
                plt.plot(
                    theta_grid[:, p_num],
                    mu,
                    self._DEFAULT_COLOR_CYCLE[i],
                    label=r"%s, $\mu$" % benchmark,
                )

                for j, z in enumerate(z_contours):
                    plt.plot(
                        theta_grid[:, p_num],
                        mu + sigma * z,
                        self._DEFAULT_COLOR_CYCLE[i],
                        linestyle=self._DEFAULT_LINESTYLE_CYCLE[j],
                        label=r"%s, $%s\sigma $" % (benchmark, z),
                    )
                    plt.plot(
                        theta_grid[:, p_num],
                        mu - sigma * z,
                        self._DEFAULT_COLOR_CYCLE[i],
                        linestyle=self._DEFAULT_LINESTYLE_CYCLE[j],
                    )
                    if fill_contours:
                        plt.fill_between(
                            theta_grid[:, p_num],
                            y1=(mu + sigma * z),
                            y2=(mu - sigma * z),
                            facecolor=self._DEFAULT_COLOR_CYCLE[i],
                            alpha=alphas[j],
                        )

            plt.legend(
                bbox_to_anchor=(0.5, bb_b + bb_m * (len(z_contours))),
                ncol=len(log_r_hat_dict),
                fancybox=True,
                loc="upper center",
            )
            plt.xlabel(r"$\theta_%s$: %s" % (p_num + 1, parameter))
            plt.ylabel(
                r"$\mathbb{E}_x [ -2\, \log \,\hat{r}(x | \theta, \theta_{SM}) ]$"
            )
            plt.tight_layout()
            plt.savefig(
                "{}/evaluation_result_param_{}.png".format(evaluation_dir, parameter),
                bbox_inches="tight",
            )
            plt.show()
        return self.error_codes.Success
