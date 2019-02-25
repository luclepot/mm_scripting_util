import mm_scripting_util as mm 

def test_check_valid_cards_function(): 
    t = mm.core.miner(
        name="valid_cards",
        loglevel=10
    )
    t.setup_cards(n_samples=123098)
    assert(t._check_valid_cards(2))

def test_cards_simple():
    t = mm.core.miner(
        name="card_test",
        path=None, 
        loglevel=10
    )
    t.setup_cards(
        n_samples=1000000, force=False
    )
    assert(t._dir_size(t.dir + "/cards") == 16)
    t.destroy_sample()

def test_cards_collision():
    t = mm.core.miner(
        name="card_test",
        path=None, 
        loglevel=10
    )
    t.setup_cards(
        n_samples=1000000, force=False
    )
    
    assert(t._dir_size(t.dir + "/cards") == 16)

    j = mm.core.miner(
        name="card_test",
        path=None,
        loglevel=10
    )

    excepted = False
    try:
        j.setup_cards(
            n_samples=500000, force=False 
        )
    except Exception:
        excepted = True
        print("j failed to setup cards!!")
    ## should be unchanged
    assert(j._dir_size(j.dir + "/cards") == 16)
    assert(excepted)
    
    j.setup_cards(
        n_samples=500000, force=True
    )

    assert(j._dir_size(j.dir + "/cards") == 11)
    j.destroy_sample()
    t.destroy_sample()

def test_cards_remove():
    for i in range(3): 
        t = mm.core.miner(
            name="multiple_test",
            path=None,
            loglevel=10
        )
        t.setup_cards(
            n_samples=400000, force=True
        )
        assert(t._dir_size(t.dir + "/cards") == 10)
    t = mm.core.miner(
        name="multiple_test",
        path=None,
        loglevel=10
    )
    t.destroy_sample()
    
    for i in range(3): 
        tp = mm.core.miner(
            name="multiple_test",
            path=None,
            loglevel=10
        )
        tp.setup_cards(
            n_samples=400000, force=False
        )
        assert(t._dir_size(t.dir + "/cards") == 10)
        tp.destroy_sample() 

def test_cards_not_base_100k():
    t = mm.core.miner(
        name="421232k_events",
        loglevel=10
    )
    t.setup_cards(n_samples=421232)
    assert(t._dir_size(t.dir + "/cards") == 5 + 6)
    assert(t._check_valid_cards(5))

    t.setup_cards(n_samples=120000, force=True)
    assert(t._dir_size(t.dir + "/cards") == 2 + 6)
    assert(t._check_valid_cards(2))

    t.setup_cards(n_samples=1000000, force=True)
    assert(t._dir_size(t.dir + "/cards") == 10 + 6)
    assert(t._check_valid_cards(10))
    t.destroy_sample()