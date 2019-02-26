import mm_scripting_util as mm

def test_simulate_basic(): 
    t = mm.core.miner(
        name="simulate_basic",
        loglevel=10,
        autodestruct=True,
        backend="tth.dat"
    )
    t.simulate_data(
        samples=10000,
        sample_benchmark='w'
    )

def test_simulate_step():
    t = mm.core.miner(
        name="simulate_basic",
        loglevel=10,
        autodestruct=True,
        backend="tth.dat"
    )
    t.setup_cards(10000)
    t.simulate_data(
        samples=10000,
        sample_benchmark='sm',
        force=False
    )