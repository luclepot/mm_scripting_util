import mm_scripting_util as mm 


def test_valid_morphing_function():
    t = mm.core.miner(
            loglevel=10
        )
    t.run_morphing(
        morphing_trials=1000,
        force=False
    )
    assert(t._check_valid_morphing())

def test_simple_collision():
    t = mm.core.miner(
            loglevel=10
        )
    j = mm.core.miner(
        loglevel=10
    )
    t.run_morphing(
        morphing_trials=1000,
        force=False
    )
    excepted = False
    try:
        j.run_morphing(
            morphing_trials=2000,
            force=False
        )
    except FileExistsError: 
        excepted = True
    assert(excepted)
    
    assert(t._dir_size(t.dir + "/data") == 1)

    j.run_morphing(
        morphing_trials=2000,
        force=True
    )

    assert(t._dir_size(t.dir + "/data") == 1)
   
    t.destroy_sample()
    j.destroy_sample()

