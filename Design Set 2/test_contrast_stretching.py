import numpy as np


def bounded_random_walk(length, lower_bound,  upper_bound, start, end, std):
    assert (lower_bound <= start and lower_bound <= end)
    assert (start <= upper_bound and end <= upper_bound)

    np.set_printoptions(threshold=np.inf)

    bounds = upper_bound - lower_bound

    rand = (std * (np.random.random(length) - 0.5)).cumsum()
    print('rand:')
    print(rand)
    rand_trend = np.linspace(rand[0], rand[-1], length)
    print('rand_trend:')
    print(rand_trend)
    rand_deltas = (rand - rand_trend)
    rand_deltas /= np.max([1, (rand_deltas.max()-rand_deltas.min())/bounds])
    print('rand_deltas:')
    print(rand_deltas)

    trend_line = np.linspace(start, end, length)
    upper_bound_delta = upper_bound - trend_line
    lower_bound_delta = lower_bound - trend_line

    upper_slips_mask = (rand_deltas-upper_bound_delta) >= 0
    print('upper_slips_mask:')
    print(upper_slips_mask)
    upper_deltas = rand_deltas - upper_bound_delta
    rand_deltas[upper_slips_mask] = (upper_bound_delta - upper_deltas)[upper_slips_mask]
    print('upper_deltas:')
    print(upper_deltas)

    lower_slips_mask = (lower_bound_delta-rand_deltas) >= 0
    print('lower_slips_mask:')
    print(lower_slips_mask)
    lower_deltas = lower_bound_delta - rand_deltas
    rand_deltas[lower_slips_mask] = (lower_bound_delta + lower_deltas)[lower_slips_mask]
    print('lower_deltas:')
    print(lower_deltas)

    return trend_line + rand_deltas


randomData = bounded_random_walk(100, 50, 100, 50, 100, 10)
print('randomData:')
print(randomData)