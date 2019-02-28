import mm_scripting_util as mm

def test_simulate_basic(): 
    t = mm.core.miner(
        name="simulate_basic",
        loglevel=10,
        autodestruct=True,
        backend="tth.dat"
    )
    ret = t.simulate_data(
        samples=10000,
        sample_benchmark='w'
    )
    assert(t.error_codes.Success in ret)

def test_simulate_step():
    t = mm.core.miner(
        name="hoot_hoot",
        loglevel=10,
        autodestruct=True,
        backend="tth.dat"
    )
    t.setup_cards(10000)
    assert(t._get_simulation_step(t._number_of_cards(10000, 100000), 10000) == 1)
    ret = t.simulate_data(
        samples=10000,
        sample_benchmark='bullshit',
        force=False
    )
    self.log.info(ret)
    assert(t.error_codes.CaughtExceptionError in ret)
