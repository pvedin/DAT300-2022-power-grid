from time import time
from powergrid_ac import *

def timeit(f):
    """
    Wrapper that can be used to measure the time taken to run a test.
    """
    def f2(*args, **kwargs):
        t1 = time()
        ret = f(*args, **kwargs)
        t2 = time()
        print("Elapsed: ", t2 - t1)
        return ret

    return f2

def println(*args, **kwargs):
    if "sep" in kwargs:
        del kwargs["sep"]
    print(*args, **kwargs, sep="\n")


def print_anomalies(net, r):
    anomalies = net.check_for_anomalies(r)
    # (row, col) == (measurement_index, timestamp)
    println("Anomalies:", [(f"ts {i[1]} index {i[0]}", r[i[0], i[1]]) for i in anomalies])
    print(len(anomalies))

@timeit
def test_healthy():
    net = PowerGrid(14)
    z = net.create_measurements(4, env_noise=False)
    Hs, x_ests, _z_x_ests = net.estimate_state(z)
    r = net.calculate_normalized_residuals(Hs, z, x_ests)
    println("r", r)
    print_anomalies(net, r)

@timeit
def test_norm_1():
    net = PowerGrid(14)
    np.random.seed(10)
    z = net.create_measurements(4, env_noise=True)

    Hs, x_ests, z_x_ests = net.estimate_state(z)

    config = {
        "net": net,
        "H": Hs[0],
        "x_est": x_ests[0],
        "z_x_est": z_x_ests[0],
        "fixed": {1: 0.01},
        "silence": True
    }

    z_a = np.copy(z)

    injection_vectors = [] # [(a,c), ...]

    for ts in range(2, z.shape[1]):
        config["H"] = Hs[ts]
        config["x_est"] = x_ests[ts]
        config["z_x_est"] = z_x_ests[ts]
        a, c = AnomalyModels.least_effort_norm_1(**config)
        injection_vectors.append((a,c))
        z_a[:, ts] += a
    Hs_a, x_ests_a, _z_x_ests_a = net.estimate_state(z_a)

    r = net.calculate_normalized_residuals(Hs_a, x_ests_a, z_a)
    println("r", r)
    print_anomalies(net, r)

if __name__ == "__main__":
    #test_healthy()
    test_norm_1()
