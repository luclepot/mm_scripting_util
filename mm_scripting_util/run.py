from mm_scripting_util.core import miner 
import subprocess
import sys
import os
import logging
import argparse
import traceback
import numpy as np
import lucs_tools

run_main = True

desc_str = \
"""MM_SCRIPTING_UTIL COMMAND LINE PROGRAM\n
The cmd line program for the mm_scripting_util class.
Gives (almost) all of the functionality of the base class,
in a single (great) command line program.
Tips:
 - if creating a NEW sample, make sure to specify NAME and BACKEND
   as the first parameters to the program. 
 - If the sample already exists, you may select it by default by 
   running the utility within the top directory of the sample.
   example: with a sample at ~/sample, cd ~/sample and run the
       utility without the NAME or BACKEND specs (automatic!!)
 - For more in-depth documentation, see the python code itself
   (which you definitely have if you're running this)\n
Fully avaliable BACKEND specifications in your current directory:\n"""

full_av = miner.list_full_backends()

for backend in full_av:
    desc_str += """ - {}\n""".format(backend)

def rng(s):
    try:
        _min, _max = map(float, s.split(','))
        return _min, _max
    except: 
        raise argparse.ArgumentTypeError(s)

def tup(s, t):
    try:
        return tuple(map(t, s.split(',')))
    except:
        raise argparse.ArgumentTypeError(s)

def write_parser(
    desc_str
):
    desc_str = lucs_tools.formatting.header.fmt(desc_str, side='c')

    module_directory = os.path.dirname(os.path.abspath(__file__))

    sd = {
        'STR' : lambda req, sname='-n', name='--name', _help=None, default=None, dest=None: [
                (sname, name), {
                    'action': 'store',
                    'dest': name.strip('-').replace('-', '_') if dest is None else dest,
                    'type': str,
                    'help': _help,
                    'required': bool(req),
                    'default': default
                }
            ],
        'NUM' : lambda ntype=int, req=True, sname='-n', name='--num', default=None, _help=None, dest=None: [
                (sname, name), {
                    'action': 'store', 
                    'dest': name.strip('-').replace('-', '_') if dest is None else dest,
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
        'TUPLE' : lambda type_, req=True, sname='-t', name='--tuple', default=(None,), _help=None: [
                (sname, name), {
                    'action': 'store',
                    'dest': name.strip('-').replace('-', '_'),
                    'type': lambda s: tup(s, t=type_),
                    'required': req,
                    'default': default,
                    'help': _help,
                }
        ]
    }

    global_options = [
        # sd['STR'](False, '-B', '--BACKEND'),
        sd['POS']('NAME', None, 'name for overall sample (required ONLY if pwd is NOT a sample)'),
        sd['POS']('BACKEND', None, 'backend; default choices of {} (required ONLY if pwd is NOT a sample)'.format(list(full_av))),
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
            sd['NUM'](int, True, '-s', '--samples'),
            sd['STR'](True, '-b', '--sample-benchmark'),
            sd['STR'](False, '-mg', '--mg-dir'),
            sd['STR'](True, '-e', '--mg-environment-cmd', _help='Enter an environment setup command; otherwise choose default options ( lxplus7 | ubc )'),
            sd['BOOL'](name='--force'),
            sd['NUM'](int, False, '-o', '--override-step', None)
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
                sd['STR'](True, '-e', '--evaluation-name'),
                sd['STR'](False, '-t', '--training-name'),
                sd['LIST'](False, '-z', '--z-contours', float),
                sd['BOOL'](True, False, '-f', '--fill-contours')
                ],
            },
        'augment' : [
            sd['STR'](True, '-n', '--sample-name'),
            sd['NUM'](int, True, '-s', '--samples', dest='n_or_frac_augmented_samples', ),
            sd['STR'](True, '-b', '--augmentation-benchmark'),
            sd['BOOL'](True, False, name='--force')
            ],
        'train' : [
            sd['STR'](True, '-s', '--sample-name'),
            sd['STR'](True, '-n', '--training-name'),
            sd['TUPLE'](int, False, '-na','--node-architecture', default=(100,100,100)),
            sd['NUM'](int, False, '-e', '--n-epochs', default=30),
            sd['NUM'](int, False, '-bs','--batch-size', default=128),
            sd['STR'](False,'-af','--activation-function', default='relu'),
            sd['STR'](False,'-t', '--trainer', default='adam'),
            sd['NUM'](float, False,'-ilr', '--initial-learning-rate', default=0.001),
            sd['NUM'](float, False,'-flr', '--final-learning-rate', default=0.001),
            sd['BOOL'](True, False, name='--force')
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

def main():

    parser = write_parser(desc_str)

    argv = sys.argv[1:]

    # if we are IN a sample directory, automatically select
    prepend = []
    in_dir = False
    if os.path.exists('config.mmconfig'):
        d = miner._load_config(os.path.abspath('config.mmconfig'))
        prepend = [d['name'], d['backend']]
        in_dir = True

    if len(argv) > 0:
        args = parser.parse_args(prepend + argv)
    else:
        parser.print_help()
        exit(0)

    m = miner(
        name=args.NAME,
        backend=args.BACKEND,
        path=os.path.dirname(os.path.abspath(os.getcwd())) if in_dir else None, 
        card_directory=args.CARD_DIRECTORY, 
        loglevel=args.LOG_LEVEL,
        madminer_loglevel=args.MADMINER_LOG_LEVEL,
        init_loglevel=10,
        _cmd_line_origin=True
        )

    cmd = args.COMMAND

    args_to_pass = { var: vars(args)[var] for var in vars(args) if var.islower()}
    if cmd=='ls':
        getattr(m, 'list_{}'.format(args.ls_type))(True, args.criteria, args.include_info)
    elif cmd=='simulate':
        m.simulate_data(**args_to_pass)
    elif cmd=='plot':
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
        m.augment_samples(**args_to_pass)
    elif cmd=='train':
        m.train_method(**args_to_pass)
    elif cmd=='evaluate':
        pass
    else:
        raise argparse.ArgumentTypeError

    return m

m = None
if run_main: 
    m = main()
