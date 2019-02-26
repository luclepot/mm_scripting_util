
from .util import *
import madminer.core


"""
Main driver container for the class. 

This class should handle all necessary interfacing with
madminer, in the context of the ttH CP process. 

"""

class miner(mm_util): 

    def __init__(
            self,
            name="temp",
            path=None,
            loglevel=logging.INFO,
            autodestruct=False,
            backend="tth.dat"
        ):
        """
        madminer-helper object for quickly running madgraph scripts. 
        parameters:
            name:
                string, the name of the main object directory
            path: 
                string, path to desired object directory. Will default to current working directory.
            loglevel:
                loglevel for tth class as a whole. Will default to INFO.
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

        self._load_backend(backend)

        self.STEP = 0

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
        except:
            self.log.error(traceback.format_exc())
            self.log.error("ABORTING")

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
            seeds = np.random.randint(1, 31000.**2., len(sample_sizes))

        # check validity of seed input (matching with sample sizes, at the least)
        assert(len(seeds) >= len(sample_sizes))

        self._check_directory(
            local_pathname="cards",
            force=force,
            pattern="card"
        )

        files = os.listdir(self.module_path + "/data/cards")

        for f in files: 
            shutil.copyfile(
                src=self.module_path + "/data/cards/" + f, 
                dst=self.dir + "/cards/" + f
            )

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
                self.params['madgraph_generation_command'] + '\n',
                ""]
        )

        self.log.debug("Setup {} cards in dir {}".format(
            len(sample_sizes), self.dir + "/cards")
        )

        files = os.listdir(self.dir + "/cards")
        for f in files: 
            self.log.debug(" - \"{}\"".format(f))

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
                    parameter_name=self.params['parameters'][parameter]['name'],
                    morphing_max_power=self.params['parameters'][parameter]['morphing_max_power'],   
                    parameter_range=self.params['parameters'][parameter]['parameter_range']
                )

        for parameter in self.params['parameters']:
            for benchmark in self.params['parameters'][parameter]['parameter_benchmarks']:
                self.madminer_object.add_benchmark(
                    {parameter:benchmark[0]},benchmark[1]
                )

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
                self._check_valid_backend(benchmark_to_check=sample_benchmark)):
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
                self.log.warning("MGDIR variable '{}' invalid")
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
    
    def set_loglevel(
            self,
            loglevel
        ):     

        logging.basicConfig(
            format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
            datefmt='%H:%M',
            level=logging.WARNING
        )

        logging.getLogger("mm_scripting_util").setLevel(loglevel)
        
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
