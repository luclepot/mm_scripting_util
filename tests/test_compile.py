import mm_scripting_util as mm 
import os 

def test_valid_init_function(): 
    t = mm.core.miner(loglevel=10)
    assert(t._check_valid_init())

    t.destroy_sample() 
    assert(not t._check_valid_init())

def test_general(): 
    t = mm.core.miner(loglevel=10, autodestruct=True)
    assert(t._check_valid_init())
    tdir = t.dir 
    assert(os.path.exists(tdir))    
    t.destroy_sample() 
    assert(not os.path.exists(tdir))
    print("directory:", tdir)

def test_named():
    t = mm.core.miner(name="test_compile_miner", path=None, loglevel=10)
    tdir = t.dir 
    assert(os.path.exists(tdir))    
    t.destroy_sample() 
    assert(not os.path.exists(tdir))
    print("directory:", tdir)

def test_with_path():
    t = mm.core.miner(name="test_compile_miner", path=os.getcwd() + "/..", loglevel=10)
    tdir = t.dir 
    assert(os.path.exists(tdir))    
    t.destroy_sample() 
    assert(not os.path.exists(tdir))
    print("directory:", tdir)
