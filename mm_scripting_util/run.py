from .core import tth_miner
import subprocess
import sys
import os
import logging

logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.INFO
)

true_strings = ["True", "1", "true", "t", "y", "Yes"]

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)

def main():
    name="temp"
    samples=200000
    sample_limit=100000
    eval_samples=5000
    architecture=(10,10,10)
    epochs=10
    generate_samples="False"
    train_samples="True"
    help_flag=False
    t_sample = None
    t_train = None

    for i,arg in enumerate(sys.argv):
        if arg=="--name" or arg=='-n':
            name=str(sys.argv[i + 1])
        elif arg=="--samples" or arg=='-s':
            samples=int(sys.argv[i + 1])
        elif arg=="--limit" or arg=='-l':
            sample_limit=int(sys.argv[i + 1])
        elif arg=="--eval" or arg=='-e':
            eval_samples=int(sys.argv[i + 1])
        elif arg=="--arch" or arg=='-a':
            architecture = tuple(int(elt) for elt in sys.argv[i + 1].split(","))
        elif arg=="--epochs" or arg=='-ep':
            epochs=int(sys.argv[i + 1])
        elif arg=="--generate" or arg=='-g':
            generate_samples=str(sys.argv[i + 1])
        elif arg=="--train" or arg=='-t':
            train_samples=str(sys.argv[i + 1])
        elif arg=="--help" or arg=="-h":
            help_flag=True
            
    if help_flag:
        print("--- testing utility `run_all.py` ---")
        print()
        print("--name, -n:")
        print("    name of sample trial. default temp")
        print("--samples, -s:")
        print("    number of training/simulation samples")
        print("--limit, -l:")
        print("    max number of samples per madgraph run")
        print("--eval, -e:")
        print("    number of samples to evaluate network on")
        print("--arch, -a:")
        print("    node architecture for nerual network")
        print("--epochs, -ep:")
        print("    epochs for NN training")
        print("--generate, -g:")
        print("    bool, whether to generate new data or to just train")
        return [None, None] 

    proc = subprocess.run(args=["pwd"], encoding='utf-8', stdout=subprocess.PIPE)
    path = proc.stdout.split('\n')[0]

    # if generate_samples in true_strings:
    #     t = tth_miner(
    #             name=name,
    #             path=path
    #         )
        
    #     print("tth sampler created") 
        
    #     t.full_sample_generate_data(
    #             samples=samples,
    #             path=path,
    #             name=name,
    #             sample_limit=sample_limit
    #         )

    #     print("sample scripts generated")
        
    #     cmd = "env -i bash -l -c 'source /cvmfs/sft.cern.ch/lcg/views/LCG_94/x86_64-centos7-gcc62-opt/setup.sh; source {}/{}/mg_processes/signal/madminer/run.sh'".format(path, name)
    #     os.system(cmd)
    #     print("samples generated")
        
    #     t.full_sample_add_observables(
    #             samples=samples,
    #             path=path,
    #             name=name,
    #             sample_limit=sample_limit
    #         )

    #     print("observables added")

    #     t.plot_madgraph_data(
    #             image_save_name=name
    #         )
    
    # if train_samples in true_strings:
    #     t_train = tth.train_sample_data_here(
    #             sample_number=samples,
    #             eval_number=eval_samples,
    #             nodes=architecture,
    #             epochs=epochs,
    #             name=name
    #         )

    print("sample data trained")    
    return [t_sample, t_train]

main()