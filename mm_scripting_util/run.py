from mm_scripting_util.core import miner
import subprocess
import sys
import os
import logging
import argparse
import traceback
import numpy as np

run_main = True

def rng(s):
    try:
        _min, _max = map(float, s.split(','))
        return _min, _max
    except: 
        raise argparse.ArgumentTypeError

class header:
    styles = ['l', 'c']
    def __init__(self, width, char, colwidth=1):
        self.output = ''
        self.width = width
        self.char = char
        self.colwidth = colwidth
        self.hline()
    
    def __call__(self, s='', side='l'):
        assert(side in self.styles)
        newline = self.char*self.colwidth
        if side is 'l':
            newline += ' '*self.colwidth
        else: 
            newline += int((self.width - 2*self.colwidth - len(s))/2.0)*' '    
        newline += s
        newline += (self.width - len(newline) - 1*self.colwidth)*' '
        newline += self.char*self.colwidth + '\n'
        self.output += newline
    
    def __str__(self):
        return self.output
    
    def hline(self):
        newline = self.char*self.width + '\n'
        self.output += newline

    @staticmethod
    def fmt(s, char='%', colwidth=2, factor=2, side='l'):
        slist = s.split('\n')
        len_max = max(*map(len, slist))
        h = header(len_max + 2*colwidth*factor, char, colwidth)
        h()
        for i,line in enumerate(slist):
            if len(line) == 0:
                h()
                h.hline()
                if i < len(slist) - 1:
                    h()
            else:
                h(line, side='l')

        return str(h)
        
def write_parser():

    desc_str = """MM_SCRIPTING_UTIL COMMAND LINE PROGRAM\n
The cmd line program for the mm_scripting_util class.
Gives (almost) all of the functionality of the base class,
in a single (great) command line program.
Tips:
 - always specify NAME, BACKEND, and then the command you'd
   like to run, otherwise you'll have a bad time
 - For more in-depth documentation, see the python code itself
   (which you definitely have documented if you're running this)
"""
    desc_str = header.fmt(desc_str, side='c')

    module_directory = os.path.dirname(os.path.abspath(__file__))

    sd = {
        'STR' : lambda req, sname='-n', name='--name', _help=None, default=None: [
                (sname, name), {
                    'action': 'store',
                    'dest': name.strip('-').replace('-', '_'),
                    'type': str,
                    'help': _help,
                    'required': bool(req),
                    'default': default
                }
            ],
        'NUM' : lambda ntype=int, req=True, sname='-n', name='--num', default=None, _help=None: [
                (sname, name), {
                    'action': 'store', 
                    'dest': name.strip('-').replace('-', '_'),
                    'type': ntype,
                    'required': bool(req),
                    'default': default,
                    'help': _help,
                }
            ],
        'RANGE' : lambda req=False, sname='-r', name='--ranges', _help=None: [
                (sname, name), {
                    'action': 'store',
                    'dest': name.strip('-').replace('-', '_'),
                    'type': rng,
                    'required': req,
                    'default': None,
                    'help': _help,
                    'nargs': '*'
                }  
            ],
        'TYPE' : lambda types, name='TYPE', _help=None: [
                (name,), {
                    'choices': types,
                    'help': _help,
                }
            ],
        'BOOL' : lambda store=True, req=False, sname='-f', name='--flag', _help=None: [
                (sname, name), {
                    'action': 'store_true' if store else 'store_false',
                    'dest': name.strip('-').replace('-', '_',),
                    'required': req,
                    'default': not store, 
                    'help': _help, 
                }
            ],
        'POS' : lambda name='POS', dispname=None, _help=None: [
                (name,), {
                    'action': 'store',
                    'metavar': name if dispname is None else dispname,
                    'help': _help,
                }
            ],
        'LIST' : lambda req=True, sname='-l', name='--list', type=int, _help=None, default=None, nargs='+': [
                (sname, name), {
                    'action': 'store',
                    'dest': name.strip('-').replace('-', '_'),
                    'type': type,
                    'help': _help,
                    'default': default,
                    'required': req,
                    'nargs': '+'
                }
            ],
    }

    global_options = [
        sd['POS']('NAME', None, 'name for overall sample'),
        # sd['STR'](False, '-N', '--NAME'),
        sd['POS']('BACKEND', None, 'backend; default choices'),
        # sd['STR'](False, '-B', '--BACKEND'),
        sd['STR'](False, '-CD', '--CARD-DIRECTORY'),
        sd['NUM'](int, False, '-LL', '--LOG-LEVEL', 20),
        sd['NUM'](int, False, '-MLL', '--MADMINER-LOG-LEVEL', 20),
        ]

    commands = {
        'ls' : [
            sd['TYPE'](['samples', 'models', 'evaluations'], 'ls_type', 'type for ls command'),
            sd['STR'](False, '-c', '--criteria', 'ls filter criteria', '*'),
            sd['BOOL'](True, False, '-i', '--include-info'),
            ],
        'simulate' : [
            # sd['STR'](True),
            sd['NUM'](sname='-n'),
            sd['STR'](True, '-b', '--benchmark'),
            sd['STR'](False, '-mg', '--mg-dir'),
            sd['STR'](False, '-e', '--env-command', default='lxplus7')
            ],
        'plot' : {
            'mg': [
                sd['STR'](False, '-n', '--image-save-name'),
                sd['NUM'](int, False, '-b', '--bins', 40),
                sd['RANGE'](False, '-r', '--ranges'),
                sd['BOOL'](True, False, '-a', '--include-automatic-benchmarks', 'display automatically selected benchmarks'),
            ],
            'aug': [
                sd['STR'](True, '-s', '--sample-name'),
                sd['STR'](False, '-n', '--image-save-name'),
                sd['NUM'](int, False, '-b', '--bins', 40),
                sd['RANGE'](False, '-r', '--ranges'),
                sd['BOOL'](True, False, '-a', '--include-automatic-benchmarks', 'display automatically selected benchmarks'),
            ],
            'comp': [
                sd['STR'](True, '-s', '--sample-name'),
                sd['STR'](False, '-n', '--image-save-name'),
                sd['BOOL'](True, False, '-o', '--mark-outlier-bins'),
                sd['NUM'](int, False, '-b', '--bins', 40),
                sd['RANGE'](False, '-r', '--ranges'),
                sd['NUM'](float, False, '-t', '--threshold', 2.0),
                sd['BOOL'](True, False, '-a', '--include-automatic-benchmarks', 'display automatically selected benchmarks'),
            ],
            'eval': [
                sd['STR'](True, '-e', '--eval-name'),
                sd['STR'](False, '-t', '--training-name'),
                sd['LIST'](False, '-z', '--z-contours', float),
                sd['BOOL'](True, False, '-f', '--fill-contours')
            ],

            },
        'augment' : [

            ],
        'train' : [

            ],
        'evaluate' : [

            ]
    }

    command_subparsers = {} 

    parser = argparse.ArgumentParser(
        description=desc_str,
        formatter_class=argparse.RawTextHelpFormatter
    )

    for argument in global_options: 
        parser.add_argument(*argument[0], **argument[1])

    subparsers = parser.add_subparsers(dest='COMMAND')

    subparsers.required = True

    for command in commands:
        command_subparsers[command] = [subparsers.add_parser(command), {}]
        if type(commands[command]) != dict:
            for argument in commands[command]:
                command_subparsers[command][0].add_argument(*argument[0], **argument[1])
        else:
            sub = command_subparsers[command][0].add_subparsers(dest='{}_TYPE'.format(command.upper()))
            sub.required = True
            for subcommand in commands[command]:
                command_subparsers[command][1][subcommand] = sub.add_parser(subcommand)
                for argument in commands[command][subcommand]:
                    command_subparsers[command][1][subcommand].add_argument(*argument[0], **argument[1])
    
    return parser


def __main__():

    parser = write_parser()
    if len(sys.argv[1:]) > 0:
        args = parser.parse_args(sys.argv[1:])
    else:
        parser.print_help()
        exit(0)

    m = miner(
        name=args.NAME,
        backend=args.BACKEND,
        card_directory=args.CARD_DIRECTORY, 
        loglevel=args.LOG_LEVEL,
        init_loglevel=10
        )

    cmd = args.COMMAND

    if cmd=='ls':
        getattr(m, 'list_{}'.format(args.ls_type))(True, args.criteria, args.include_info)
    elif cmd=='simulate':
        pass
    elif cmd=='plot':
        args_to_pass = {var: vars(args)[var] for var in vars(args) if not var.isupper()}
        print(args_to_pass)
        if args.PLOT_TYPE=='mg':
            m.plot_mg5_data_corner(**args_to_pass)
        elif args.PLOT_TYPE=='aug':
            m.plot_augmented_data_corner(**args_to_pass)
        elif args.PLOT_TYPE=='comp':
            m.plot_compare_mg5_and_augmented_data(**args_to_pass)
        elif args.PLOT_TYPE=='eval':
            m.plot_evaluation_results(**args_to_pass)
        else:
            raise argparse.ArgumentTypeError 
    elif cmd=='augment':
        pass
    elif cmd=='train':
        pass
    elif cmd=='evaluate':
        pass
    else:
        raise argparse.ArgumentTypeError

    return 0
    ## CMD, positional argument
    parser.add_argument("cmd", 
            nargs=1,
            choices=commands
        )

    ## required initilization arguments
    parser.add_argument("-n", "--name",
        action="store",
        dest="name",
        type=str,
        help="master simulation folder name",
        required=True
        )
    parser.add_argument("-b", "--backend",
        action="store",
        dest="backend",
        type=str,
        required=True,
        help="backend name or path",
        )

    
    cmdargs, unknown = parser.parse_known_args(sys.argv[1:])
    
    ## general arguments
    # name


    parser.add_argument("-l","--log-level",
        action="store",
        dest="loglevel",
        default=10,
        type=int,
        help="loglevel for miner object",
        )

    parser.add_argument("-f","--force",
        action="store_true",
        dest="force",
        default=False,
        help="boolean for forcing overwrite of prev. data sims",
        )
    parser.add_argument("-r","--runtime",
        action="store",
        dest="max_runtime",
        default=60 * 60,
        type=int,
        help="max runtime in seconds of process (for condor/flashy)",
        )
    parser.add_argument("-rf","--run-flashy",
        action="store_true",
        dest="run_flashy",
        default=False,
        help="boolean for running script on flashy server batch system",
        )
    parser.add_argument("-bins","--bins",
        action="store",
        dest="bins",
        default=(40, 40),
        type=int,
        help="tuple of bins",
        nargs="+",
        )

    ## simulation related arguments
    parser.add_argument("-s","--sim",
        action="store_true",
        dest="generate",
        default=False,
        help="boolean for data simulation",
        )
    parser.add_argument("-ss","--sim-samples",
        action="store",
        dest="samples",
        default=100000,
        type=int,
        help="number of data simulation samples",
        )
    parser.add_argument("-sb","--sim-benchmark",
        action="store",
        dest="sample_benchmark",
        default="sm",
        type=str,
        help="sample benchmark at which to generate data",
        )
    parser.add_argument("-sd","--sim-card-directory",
        action="store",
        dest="card_directory",
        default=None,
        type=str,
        help="path to card directory",
        )
    parser.add_argument("-sp","--sim-use-pythia",
        action="store_true",
        dest="use_pythia_card",
        default=False,
        help="boolean for using pythia card in simulation",
        )
    parser.add_argument("-sstep","--sim-step",
        action="store",
        dest="simulation_step",
        default=0,
        type=int,
        help="simulation step to start at",
        )

    ## training related arguments
    parser.add_argument("-t","--train",
        action="store_true",
        dest="train",
        default=False,
        help="boolean, flags whether or not to train the given data",
        )
    parser.add_argument("-ts","--train-samples",
        action="store",
        dest="augmented_samples",
        default=100000,
        type=int,
        help="number of augmented samples to draw, using madminer's sample augmenter",
        )
    parser.add_argument("-tn","--train-name",
        action="store",
        dest="training_name",
        default="temp",
        type=str,
        help="name for training/augmented samples draw",
        )
    parser.add_argument("-tb","--train-benchmark",
        action="store",
        dest="augmentation_benchmark",
        default="sm",
        type=str,
        help="sample benchmark at which to train/augment data",
        )
    parser.add_argument("-tstep","--train-step",
        action="store",
        dest="training_step",
        default=0,
        type=int,
        help="training step to start at",
        )
    parser.add_argument("-ti","--train-imshow",
        action="store_true",
        dest="train_imshow",
        default=False,
        help="boolean, flags whether to show or save the resultant plots",
        )

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
            card_directory=args.card_directory,
        )

        miner_object._submit_flashy(
            arg_list=[
                arg for arg in sys.argv[1:] if arg not in ["-rf", "--run-flashy"]
            ],
            max_runtime=args.max_runtime,
        )
        return 0

    ## init object
    miner_object = miner(
        name=args.name,
        loglevel=args.loglevel,
        backend=args.backend,
        card_directory=args.card_directory,
    )
    ## if generation flag, run generation function
    if args.generate:
        miner_object._exec_wrap(miner_object.simulate_data)(
            samples=args.samples,
            sample_benchmark=args.sample_benchmark,
            force=args.force,
            use_pythia_card=args.use_pythia_card,
            override_step=args.simulation_step,
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
            image_save_name=imgsv,
        )

if run_main: 
    __main__()
