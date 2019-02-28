import mm_scripting_util as mm

def test_simple():
    t = mm.core.miner(
        name="training",
        loglevel=10,
        autodestruct=False
    )

    ret = t.augment_samples(
        training_name="train_boyz",
        n_or_frac_augmented_samples=1., 
        augmentation_benchmark='w'
    )

    assert(t.error_codes.Success in ret)

def test_bad_benchmark(): 
    t = mm.core.miner(
        name="training",
        loglevel=10,
        autodestruct=False
    )

    ret = t.augment_samples(
        training_name="train_boyz",
        n_or_frac_augmented_samples=1., 
        augmentation_benchmark='bad_mon'
    )
    print(ret) 
    assert(t.error_codes.InvalidInputError in ret)