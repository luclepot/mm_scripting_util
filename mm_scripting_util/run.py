from .core import miner
import subprocess
import sys
import os
import logging
import argparse
import traceback

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
parser.add_argument('-c', '--condor',
                    action='store_true', dest='run_condor',
                    default=False,
                    help="boolean for running script on condor")

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
                    action='store', dest='custom_card_directory', 
                    default=None, type=str,
                    help="path to custom card directory")
parser.add_argument('-sp','--sim-use-pythia', 
                    action='store_true', dest='use_pythia_card',
                    default=False,
                    help="boolean for using pythia card in simulation")

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

## parse all arguments
args = parser.parse_args(sys.argv[1:])

## init object
miner_object = miner(
        name=args.name,
        loglevel=args.loglevel,
        backend=args.backend,
        custom_card_directory=args.custom_card_directory
    )

## if condor flag, write condor script calling another instance of itself (without condor flag)
if args.run_condor: 
    miner_object._submit_condor(arg_list=sys.argv)

## if generation flag, run generation function
if args.generate:
    miner_object._exec_wrap(miner_object.simulate_data)(
            samples=args.samples,
            sample_benchmark=args.sample_benchmark,
            force=args.force,
            use_pythia_card=args.use_pythia_card
        )

## if train flag, run training function
if args.train:
    miner_object._exec_wrap(miner_object.train_data)(
            augmented_samples=args.augmented_samples,
            training_name=args.training_name, 
            augmentation_benchmark=args.augmentation_benchmark
        )
