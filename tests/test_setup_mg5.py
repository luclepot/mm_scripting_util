import mm_scripting_util as mm 
import platform 
import getpass

def test_check_mg5_path_function():     
    t = mm.core.miner(
        loglevel=10,
        autodestruct=True
    )
    t.setup_cards(10000)
    assert( t.error_codes.Success != t._check_valid_mg5_scripts(10000))

    # linux only 
    if getpass.getuser() != 'llepotti':
        return

    t.run_morphing()
    ret = t.setup_mg5_scripts(
        10000, sample_benchmark='sm', mg_dir=None
    )
    assert(t.error_codes.Success in ret)
    assert(t.error_codes.Success == t._check_valid_mg5_scripts(10000))

def test_bad_mg5path():
    t = mm.core.miner(
        name="temp_mon",
        loglevel=10,
        autodestruct=True
    )
    t.setup_cards(100)
    t.run_morphing()
    t.setup_mg5_scripts(
        100, sample_benchmark='sm', mg_dir="obviously_trash_path"
    )
    assert(t.error_codes.Success != t._check_valid_mg5_scripts(samples=100))

def test_setup_mg5_clash():
    t = mm.core.miner(
        name="mg5setup",
        path=None,
        loglevel=10,
        autodestruct=True
    )

    t.setup_cards(400000)
    t.run_morphing()

    ret = t.setup_mg5_scripts(
        400000, sample_benchmark='sm',
    )

    if getpass.getuser() != 'llepotti':
        assert(len(ret) != 0)
        return

    assert(t.error_codes.Success in ret)
    excepted=False
    try:
        t.setup_mg5_scripts(
            400000, sample_benchmark='w'
        )
    except FileExistsError:
        excepted = True
    assert(excepted)
