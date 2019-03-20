from .core import miner
import subprocess
import sys
import os
import logging
import argparse
import traceback

def __main__():
    parser = argparse.ArgumentParser(description="processing for mm_scripting_util.run.py file")

    ## general arguments
    parser.add_argument('-n', '--name',
                        action='store', dest='name',
                        default="temp", type=str,
                        help="master simulation folder name")
    parser.add_argument('-l', '--log-level', 
                        action='store', dest='loglevel', 
                        default=20, type=int,
                        help="loglevel for miner object")
    parser.add_argument('-b', '--backend',
                        action='store', dest='backend',
                        default="tth.dat", type=str,
                        help="backend name or path")
    parser.add_argument('-f', '--force',
                        action='store_true', dest='force',
                        default=False,
                        help="boolean for forcing overwrite of prev. data sims")
    parser.add_argument('-r', '--runtime',
                        action='store', dest='max_runtime',
                        default=60*60, type=int,
                        help="max runtime in seconds of process (for condor/flashy)")
    # parser.add_argument('-rc', '--run-condor',
    #                     action='store_true', dest='run_condor',
    #                     default=False,
    #                     help="boolean for running script on condor server batch system")
    parser.add_argument('-rf', '--run-flashy',
                        action='store_true', dest='run_flashy',
                        default=False,
                        help="boolean for running script on flashy server batch system")
    parser.add_argument('-bins','--bins',
                        action='store', dest='bins',
                        default=(40,40), type=int, 
                        help="tuple of bins", nargs='+')

    ## simulation related arguments
    parser.add_argument('-s', '--sim',
                        action='store_true', dest='generate', 
                        default=False, 
                        help="boolean for data simulation")
    parser.add_argument('-ss','--sim-samples',
                        action='store', dest='samples', 
                        default=100000, type=int, 
                        help="number of data simulation samples")
    parser.add_argument('-sb','--sim-benchmark', 
                        action='store', dest='sample_benchmark', 
                        default='sm', type=str, 
                        help="sample benchmark at which to generate data")
    parser.add_argument('-sd','--sim-card-directory',
                        action='store', dest='card_directory', 
                        default=None, type=str,
                        help="path to card directory")
    parser.add_argument('-sp','--sim-use-pythia', 
                        action='store_true', dest='use_pythia_card',
                        default=False,
                        help="boolean for using pythia card in simulation")
    parser.add_argument('-sstep','--sim-step',
                        action='store', dest='simulation_step', 
                        default=0, type=int, 
                        help="simulation step to start at")

    ## training related arguments
    parser.add_argument('-t', '--train', 
                        action='store_true', dest='train',
                        default=False,
                        help="boolean, flags whether or not to train the given data")
    parser.add_argument('-ts','--train-samples',
                        action='store', dest='augmented_samples', 
                        default=100000, type=int, 
                        help="number of augmented samples to draw, using madminer's sample augmenter")
    parser.add_argument('-tn','--train-name',
                        action='store', dest='training_name', 
                        default='temp', type=str, 
                        help="name for training/augmented samples draw")
    parser.add_argument('-tb','--train-benchmark',
                        action='store', dest='augmentation_benchmark', 
                        default='sm', type=str, 
                        help="sample benchmark at which to train/augment data")
    parser.add_argument('-tstep','--train-step',
                        action='store', dest='training_step', 
                        default=0, type=int, 
                        help="training step to start at")
    parser.add_argument('-ti', '--train-imshow',
                        action='store_true', dest='train_imshow',
                        default=False,
                        help="boolean, flags whether to show or save the resultant plots")

    ## parse all arguments
    args = parser.parse_args(sys.argv[1:])

    if args.simulation_step == 0:
        args.simulation_step = None
    if args.training_step == 0:
        args.training_step = None

    ## if condor/flashy flag, write script calling another instance of itself (without given flags of course)
    if args.run_flashy:
        # init object with silent logging
        miner_object = miner(
            name=args.name,
            loglevel=logging.ERROR,
            backend=args.backend,
            card_directory=args.card_directory
        )   

        miner_object._submit_flashy(
                arg_list=[arg for arg in sys.argv[1:] if arg not in ["-rf", "--run-flashy"]],
                max_runtime=args.max_runtime
            )
        return 0
    
    ## init object
    miner_object = miner(
        name=args.name,
        loglevel=args.loglevel,
        backend=args.backend,
        card_directory=args.card_directory
    )   
    ## if generation flag, run generation function
    if args.generate:
        miner_object._exec_wrap(miner_object.simulate_data)(
                samples=args.samples,
                sample_benchmark=args.sample_benchmark,
                force=args.force,
                use_pythia_card=args.use_pythia_card,
                override_step=args.simulation_step
            )

    ## if train flag, run training function
    if args.train:
        if args.train_imshow:
            imgsv = None
        else: 
            imgsv = "{}_{}".format(args.training_name, args.augmentation_benchmark)
        miner_object._exec_wrap(miner_object.train_data)(
                augmented_samples=args.augmented_samples,
                training_name=args.training_name, 
                augmentation_benchmark=args.augmentation_benchmark,
                override_step=args.training_step,
                bins=tuple(args.bins),
                image_save_name=imgsv
            )

__main__()