import mm_scripting_util as mm 

def test_run_basic(): 
    t = mm.core.miner(
        name="mg5_run_small",
        loglevel=10,
        autodestruct=False
    )
    t.setup_cards(
        n_samples=1000,
        force=True
    )
    t.run_morphing()
    t.setup_mg5_scripts(
        sample_benchmark='sm',
        samples=1000,
        force=True
    )
    t.run_mg5_script(
        samples=1000,
        platform="lxplus7",
        force=True
    )
    assert(t._check_valid_mg5_run(
        samples=1000
    ))
