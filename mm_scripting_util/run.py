from .core import miner
import subprocess
import sys
import os
import logging
import argparse
import traceback

parser = argparse.ArgumentParser(description="processing for mm_scripting_util.run.py file")

## add simulation related arguments
parser.add_argument('-n', '--name',
                    action='store', dest='name',
                    default="temp", type=str,
                    help="master simulation folder name")
parser.add_argument('-s', '--samples',
                    action='store', dest='samples', 
                    default=100000, type=int, 
                    help="number of data simulation samples")
parser.add_argument('-g', '--generate',
                    action='store_true', dest='generate', 
                    default=False, 
                    help="boolean for data simulation")
parser.add_argument('-sb','--sample-benchmark', 
                    action='store', dest='sample_benchmark', 
                    default='sm', type=str, 
                    help="sample benchmark at which to generate data")
parser.add_argument('-ll','--log-level', 
                    action='store', dest='loglevel', 
                    default=20, type=int,
                    help="loglevel for miner object")
parser.add_argument('-ccd','--custom-card-directory',
                    action='store', dest='custom_card_directory', 
                    default="", type=str,
                    help="path to custom card directory")
parser.add_argument('-be','--backend',
                    action='store', dest='backend',
                    default="tth.dat", type=str,
                    help="backend name or path")
parser.add_argument('-f', '--force',
                    action='store_true', dest='force',
                    default=False,
                    help="boolean for forcing overwrite of prev. data sims")
parser.add_argument('-up', '--use-pythia', 
                    action='store_true', dest='use_pythia_card',
                    default=False,
                    help="boolean for using pythia card in simulation")
parser.add_argument('-m')
# add training related arguments
parser.add_argument('-t', '--train', 
                    action='store_true', dest='train',
                    default=False,
                    help="boolean, flags whether or not to train the given data")

args = parser.parse_args([arg for arg in sys.argv if arg is not "-m"])

miner_object = miner(
    name=args.name,
    loglevel=args.loglevel,
    backend=args.backend,
    custom_card_directory=args.custom_card_directory
)

try:
    if args.generate:
        miner_object.simulate_data(
            samples=args.samples,
            sample_benchmark=args.sample_benchmark,
            force=args.force,
            use_pythia_card=args.use_pythia_card
        )
except:
    traceback.format_exc()

try:
    if args.train:
        pass
except:
    traceback.format_exc()

# def old():
#     return 0 
#     name="temp"
#     samples=200000
#     sample_benchmark='sm'
#     eval_samples=5000
#     architecture=(10,10,10)
#     epochs=10
#     generate_samples="False"
#     train_samples="True"
#     help_flag=False
#     t_sample = None
#     t_train = None

#     for i,arg in enumerate(sys.argv):
#         if arg=="--name" or arg=='-n':
#             name=str(sys.argv[i + 1])
#         elif arg=="--samples" or arg=='-s':
#             samples=int(sys.argv[i + 1])
#         elif arg=="--limit" or arg=='-l':
#             sample_limit=int(sys.argv[i + 1])
#         elif arg=="--eval" or arg=='-e':
#             eval_samples=int(sys.argv[i + 1])
#         elif arg=="--arch" or arg=='-a':
#             architecture = tuple(int(elt) for elt in sys.argv[i + 1].split(","))
#         elif arg=="--epochs" or arg=='-ep':
#             epochs=int(sys.argv[i + 1])
#         elif arg=="--generate" or arg=='-g':
#             generate_samples=str(sys.argv[i + 1])
#         elif arg=="--train" or arg=='-t':
#             train_samples=str(sys.argv[i + 1])
#         elif arg=="--help" or arg=="-h":
#             help_flag=True
#         elif arg=="--sample-benchmark" or arg=="-sb":
#             sample_benchmark = sys.argv[i + i]
        
            
#     if help_flag:
#         print("--- command line utility `run.py` ---")
#         print()
#         print("--name, -n:")
#         print("    name of sample trial. default temp")
#         print("--samples, -s:")
#         print("    number of training/simulation samples")
#         print("--limit, -l:")
#         print("    max number of samples per madgraph run")
#         print("--eval, -e:")
#         print("    number of samples to evaluate network on")
#         print("--arch, -a:")
#         print("    node architecture for nerual network")
#         print("--epochs, -ep:")
#         print("    epochs for NN training")
#         print("--generate, -g:")
#         print("    bool, whether to generate new data or to just train")
#         return [None, None] 
