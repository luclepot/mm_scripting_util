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

def rng(s):
    try:
        _min, _max = map(float, s.split(','))
        return _min, _max
    except: 
        raise argparse.ArgumentTypeError

def write_parser():

    full_av = miner.list_full_backends()

    desc_str = """MM_SCRIPTING_UTIL COMMAND LINE PROGRAM\n
The cmd line program for the mm_scripting_util class.
Gives (almost) all of the functionality of the base class,
in a single (great) command line program.
Tips:
 - always specify NAME, BACKEND, and then the command you'd
   like to run, otherwise you'll have a bad time
 - For more in-depth documentation, see the python code itself
   (which you definitely have documented if you're running this)\n
Fully avaliable BACKEND specifications in your current directory:\n"""
    
    for backend in full_av:
        desc_str += """ - {}\n""".format(backend)

    desc_str = lucs_tools.formatting.header.fmt(desc_str, side='c')

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
            sd['NUM'](int, True, '-s', '--samples'),
            sd['STR'](True, '-b', '--sample-benchmark'),
            sd['STR'](False, '-mg', '--mg-dir'),
            sd['STR'](True, '-ec', '--mg-environment-cmd', _help='Enter an environment setup command; otherwise choose default options ( lxplus7 | ubc )'),
            sd['STR'](True, '-rc', '--mg-run-cmd', _help='Enter an environment run command; otherwise choose default options ( lxplus7 | ubc )'),
            sd['BOOL'](name='--force'),
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

def main():

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
        init_loglevel=10,
        _cmd_line_origin=True
        )

    cmd = args.COMMAND

    if cmd=='ls':
        getattr(m, 'list_{}'.format(args.ls_type))(True, args.criteria, args.include_info)
    elif cmd=='simulate':
        args_to_pass = { var: vars(args)[var] for var in vars(args) if not var.isupper()}
        m.simulate_data(**args_to_pass)
    elif cmd=='plot':
        args_to_pass = {var: vars(args)[var] for var in vars(args) if not var.isupper()}
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

    return m

m = None
if run_main: 
    m = main()
