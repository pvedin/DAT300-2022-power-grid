from os.path import isfile
from random import uniform, randint, sample
from pickle import dump, load
from re import A
from time import time

import numpy as np
from powergrid_dc import PowerGrid, AnomalyModels

import matplotlib.pyplot as plt

def timeit(f):
    """
    Wrapper that can be used to measure the time taken to run a test.
    Changes the return value to (original return values, time elapsed)
    """
    def f2(*args, **kwargs):
        t1 = time()
        ret = f(*args, **kwargs)
        t2 = time()
        elapsed = t2 - t1
        return (ret, elapsed)

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

def unique_filename(filename):
    """
    Ensures there does not exist a file with the given name in the current
    working directory. Appends a number to the filename (before the extension) 
    if such a file exists and increments it as needed, starting from 2.
    """
    ext = ""
    if "." in filename:
        i_dot = len(filename) - filename[::-1].index(".") - 1
        ext = filename[i_dot:]
        filename = filename[:i_dot]
    out = filename + ext
    count = 2
    while isfile(out):
        out = f'{filename}-{count}{ext}'
        count += 1
    return out

def pickle_dump(filepath, data):
    """
    Dumps data to a new file. Creates a new file if the given filepath already
    exists (see unique_filename()).
    """
    fn = unique_filename(filepath)
    with open(fn, "wb") as f:
        dump(data, f)

def pickle_load(fp):
    """
    Returns the contents of a pickled file.
    """
    with open(fp, "rb") as f:
        return load(f)

def generate_incomplete_knowledge_vectors(start, duration, 
        anomaly_model=AnomalyModels.random_matrix, **cfg):
    """
    anomaly_model is either random_matrix or modal_decomposition
    """
    Z = cfg["Z"].copy()
    aa = []
    for ts in range(start, start+duration):
        cfg["Z"] = Z[:ts]
        cfg["t"] = ts
        aa.append(anomaly_model(**cfg))
    
    return aa

def norm_1_sensors(secure=[]):
    print("Demo: norm-1 attack")
    net = PowerGrid(4)
    config = {
        "H": net.H,
        "secure": secure,
        "fixed": {},
        "silence": True
    }
    successes = 0
    sensor_count = {i:0 for i in range(4)}
    for k in range(4): # measurement vector contains four elements
        print(k)
        for alpha in np.linspace(-10, 10, 100):
            config["fixed"] = {k: alpha}
            a = AnomalyModels.least_effort_norm_1(**config)
            for i in np.nonzero(a)[0]:
                sensor_count[i] += 1

    print(successes)
    return sensor_count

def demo_norm_1():
    print("Demo: least_effort_norm_1")
    net = PowerGrid(4) 
    z = net.create_measurements(30, 1)
    z_a = z.copy()
    target_sensor = 1

    # time 0-4: no injection

    # time 5-9: naive injection
    a_naive = []
    for i in range(5,10):
        a = i / 10 # [0.5, 0.9]
        z_a[target_sensor, i] += a
        a_naive.append(a)

    # time 10-14: no injection

    # time 15-19: least_effort_norm_1
    config = {
        "H": net.H,
        "fixed": {target_sensor: 0.5},
    }
    a_norm_1_1 = []
    for i in range(5,10):
        config["fixed"][target_sensor] = i / 10 # [0.5, 0.9]
        a = AnomalyModels.least_effort_norm_1(**config)
        z_a[:, 10+i] += a
        a_norm_1_1.append(a)

    # time 20-24: no injection

    # time 25-29: least_effort_norm_1, with 1 secured sensor
    sensor_freq_1 = norm_1_sensors()
    secured_sensor = sorted(sensor_freq_1.items(), key=lambda t:t[1])[-1][0]
    config["secure"] = [secured_sensor]
    sensor_freq_2 = norm_1_sensors(config["secure"])
    a_norm_1_2 = []
    for i in range(5, 10):
        config["fixed"][target_sensor] = i / 10 # [0.5, 0.9]
        a = AnomalyModels.least_effort_norm_1(**config) # zero vector if infeasible
        z_a[:, 20+i] += a 
        a_norm_1_2.append(a)

    x_est = net.estimate_state(z)
    x_est_a = net.estimate_state(z_a)
    r = net.calculate_normalized_residuals(z, x_est)
    r_a = net.calculate_normalized_residuals(z_a, x_est_a) 
    print_anomalies(net, r)
    print_anomalies(net, r_a)

    print(a_norm_1_1)
    print(a_norm_1_2)

    print(sensor_freq_1)
    print(sensor_freq_2)

    td = {
        "z": z,
        "z_a": z_a,
        "x_est": x_est,
        "x_est_a": x_est_a,
        "r": r,
        "r_a": r_a,
        "naive_ts": list(range(5,10)),
        "norm_1_ts": list(range(15,20)),
        "norm_1_secure_ts": list(range(25,30)),
        "target_sensor": target_sensor,
        "secured_sensor": secured_sensor,
        "sensor_freq_1": sensor_freq_1,
        "sensor_freq_2": sensor_freq_2,
    }
    pickle_dump("data/dc_norm1_demo", td)

    print("Expected: anomalies only for ts 5-9, different perturbation for ts 25-29")
    return td

def exp_norm1():
    net = PowerGrid(14)
    z = net.create_measurements(30, 1)
    z2 = np.copy(z)
    z3 = np.copy(z) #naive

    inject_start = 10
    duration = 10
    target_sensor = 1
    alpha = 0.1
    
    z3[target_sensor, inject_start:inject_start+duration] += alpha

    config = {
        "H": net.H,
        "fixed": {target_sensor : alpha},
    }

    aa = [AnomalyModels.least_effort_norm_1(**config)] * duration

    for ts in range(len(aa)):
        z2[:, inject_start+ts] += aa[ts].transpose()
    
    x_est = net.estimate_state(z)
    x2_est = net.estimate_state(z2)
    x3_est = net.estimate_state(z3)

    r = net.calculate_normalized_residuals(z, x_est)
    r2 = net.calculate_normalized_residuals(z2, x2_est)
    r3 = net.calculate_normalized_residuals(z3, x3_est)

    r[r == np.inf] = 0
    r2[r2 == np.inf] = 0
    r3[r3 == np.inf] = 0

    td = {
        "zs": (z, z2, z3),
        "x_ests": (x_est, x2_est, x3_est),
        "rs": (r, r2, r3),
        "aa": aa
    }

    pickle_dump("data/dc_norm1", td)
    return td

def exp_small_ubiquitous():
    net = PowerGrid(14)
    z = net.create_measurements(30, 1)
    z2 = np.copy(z)

    inject_start = 10
    duration = 10
    target_sensor = 1
    alpha = 0.1

    config = {
        "H": net.H,
        "fixed": {target_sensor : alpha},

    }

    aa = [AnomalyModels.small_ubiquitous(**config)] * duration

    for ts in range(len(aa)):
        z2[:, inject_start+ts] += aa[ts].transpose()
    
    x_est = net.estimate_state(z)
    x2_est = net.estimate_state(z2)
    
    r = net.calculate_normalized_residuals(z, x_est)
    r2 = net.calculate_normalized_residuals(z2, x2_est)

    r[r == np.inf] = 0
    r2[r2 == np.inf] = 0

    td = {
        "zs": (z, z2),
        "x_ests": (x_est, x2_est),
        "rs": (r, r2),
        "aa": aa
    }

    pickle_dump("data/dc_small_ubiquitous", td)
    return td

def exp_targeted_norm1():
    net = PowerGrid(14)
    z = net.create_measurements(30, 1)
    z2 = np.copy(z)

    inject_start = 10
    duration = 10

    config = {
        "H": net.H,
        "fixed": {1 : 0.1,
                  2 : 0,
                  3 : 0.1,
                  4 : 0},
    }

    aa = [AnomalyModels.targeted_least_effort_norm_1(**config)] * duration

    for ts in range(len(aa)):
        z2[:, inject_start+ts] += aa[ts].transpose()
    
    x_est = net.estimate_state(z)
    x2_est = net.estimate_state(z2)
    
    r = net.calculate_normalized_residuals(z, x_est)
    r2 = net.calculate_normalized_residuals(z2, x2_est)

    r[r == np.inf] = 0
    r2[r2 == np.inf] = 0

    td = {
        "zs": (z, z2),
        "x_ests": (x_est, x2_est),
        "rs": (r, r2),
        "aa": aa
    }

    pickle_dump("data/dc_targeted_norm1", td)
    return td

def exp_targeted_small_ubiquitous():
    net = PowerGrid(14)
    z = net.create_measurements(30, 1)
    z2 = np.copy(z)

    inject_start = 10
    duration = 10
    
    config = {
        "H": net.H,
        "fixed": {1 : 0.1,
                  2 : 0,
                  3 : 0.1,
                  4 : 0},
    }

    aa = [AnomalyModels.targeted_small_ubiquitous(**config)] * duration

    for ts in range(len(aa)):
        z2[:, inject_start+ts] += aa[ts].transpose()
    
    x_est = net.estimate_state(z)
    x2_est = net.estimate_state(z2)

    print("a", aa[0])
    print("c", x2_est[:,15] - x_est[:,15])
    
    r = net.calculate_normalized_residuals(z, x_est)
    r2 = net.calculate_normalized_residuals(z2, x2_est)

    r[r == np.inf] = 0
    r2[r2 == np.inf] = 0

    td = {
        "zs": (z, z2),
        "x_ests": (x_est, x2_est),
        "rs": (r, r2),
        "aa": aa
    }

    pickle_dump("data/dc_targeted_small_ubiquitous", td)
    return td

def exp_targeted_matching_pursuit():
    net = PowerGrid(14)
    z = net.create_measurements(30, 1)
    z2 = np.copy(z)

    inject_start = 10
    duration = 10
    
    config = {
        "H": net.H,
        "fixed": {1 : 0.1,
                  2 : 0,
                  3 : 0.1,
                  4 : 0},
    }

    aa = [AnomalyModels.targeted_matching_pursuit(**config)] * duration

    for ts in range(len(aa)):
        z2[:, inject_start+ts] += aa[ts].transpose()
    
    x_est = net.estimate_state(z)
    x2_est = net.estimate_state(z2)

    print("a", aa[0])
    print("c", x2_est[:,10] - x_est[:,10])
    
    r = net.calculate_normalized_residuals(z, x_est)
    r2 = net.calculate_normalized_residuals(z2, x2_est)

    r[r == np.inf] = 0
    r2[r2 == np.inf] = 0

    td = {
        "zs": (z, z2),
        "x_ests": (x_est, x2_est),
        "rs": (r, r2),
        "aa": aa
    }

    pickle_dump("data/dc_targeted_matching_pursuit", td)
    return td

def exp_random_matrix_sc2():
    net = PowerGrid(14)
    z = net.create_measurements(1300, 1)
    z2 = np.copy(z)

    config = {
        "H": net.H,
        "T": 1000,
        "Z": z,
        "state_noise": net.state_noise_factor,
        "scenario": 2,
    }

    inject_start = 1100
    duration = 100
    aa = generate_incomplete_knowledge_vectors(inject_start, duration, 
            AnomalyModels.random_matrix, **config)

    for ts in range(len(aa)):
        z2[:, inject_start+ts] += aa[ts].transpose()

    x_est = net.estimate_state(z)
    x2_est = net.estimate_state(z2)

    r = net.calculate_normalized_residuals(z, x_est)
    r2 = net.calculate_normalized_residuals(z2, x2_est)

    r[r == np.inf] = 0
    r2[r2 == np.inf] = 0
    r[np.isnan(r)] = 0
    r2[np.isnan(r2)] = 0

    td = {
        "zs": (z, z2),
        "x_ests": (x_est, x2_est),
        "rs": (r, r2),
        "aa": aa
    }

def exp_random_matrix_general(sc):
    net = PowerGrid(14)
    z = net.create_measurements(1300, 1)
    z2 = np.copy(z)

    config = {
        "H": net.H,
        "T": 1000,
        "Z": z,
        "state_noise": net.state_noise_factor,
        "scenario": sc,
        "tau": 50
    }

    inject_start = 1100
    duration = 100
    aa = generate_incomplete_knowledge_vectors(inject_start, duration, 
            AnomalyModels.random_matrix, **config)

    for ts in range(len(aa)):
        z2[:, inject_start+ts] += aa[ts].transpose()

    x_est = net.estimate_state(z)
    x2_est = net.estimate_state(z2)

    r = net.calculate_normalized_residuals(z, x_est)
    r2 = net.calculate_normalized_residuals(z2, x2_est)

    r[r == np.inf] = 0
    r2[r2 == np.inf] = 0
    r[np.isnan(r)] = 0
    r2[np.isnan(r2)] = 0

    td = {
        "zs": (z, z2),
        "x_ests": (x_est, x2_est),
        "rs": (r, r2),
        "aa": aa,
        "tau": config["tau"]
    }

    pickle_dump(f"data/dc_random_matrix_sc{sc}", td)
    return td

def exp_random_matrix_sc1a():
    return exp_random_matrix_general("1a")

def exp_random_matrix_sc1b():
    return exp_random_matrix_general("1b")

def exp_random_matrix_sc2():
    return exp_random_matrix_general("2")

def exp_modal_decomposition():
    net = PowerGrid(14)
    z = net.create_measurements(1300, 1)
    z2 = np.copy(z)

    config = {
        "H": net.H,
        "T": 1000,
        "Z": z,
        "sigma": 10
    }

    inject_start = 1100
    duration = 100
    aa = generate_incomplete_knowledge_vectors(inject_start, duration, 
            AnomalyModels.modal_decomposition, **config)

    for ts in range(len(aa)):
        z2[:, inject_start+ts] += aa[ts].transpose()

    x_est = net.estimate_state(z)
    x2_est = net.estimate_state(z2)

    r = net.calculate_normalized_residuals(z, x_est)
    r2 = net.calculate_normalized_residuals(z2, x2_est)

    r[r == np.inf] = 0
    r2[r2 == np.inf] = 0
    r[np.isnan(r)] = 0
    r2[np.isnan(r2)] = 0

    td = {
        "zs": (z, z2),
        "x_ests": (x_est, x2_est),
        "rs": (r, r2),
        "aa": aa,
    }

    pickle_dump("data/dc_modal_decomposition", td)
    return td

def exp_random_noise_norm1():
    """
    Generate measurements + norm1 injections so that random noise can be
    added in exp_random_noise_norm1_p2.
    """
    net = PowerGrid(14)
    z = net.create_measurements(30, 1)
    z2 = np.copy(z)

    inject_start = 10
    duration = 10
    target_sensor = 11
    alpha = 0.2


    config = {
        "H": net.H,
        "fixed": {target_sensor : alpha},
    }

    aa = [AnomalyModels.least_effort_norm_1(**config)] * duration

    for ts in range(len(aa)):
        z2[:, inject_start+ts] += aa[ts].transpose()
    
    x_est = net.estimate_state(z)
    x2_est = net.estimate_state(z2)

    r = net.calculate_normalized_residuals(z, x_est)
    r2 = net.calculate_normalized_residuals(z2, x2_est)

    r[r == np.inf] = 0
    r2[r2 == np.inf] = 0

    td = {
        "zs": (z, z2),
        "x_ests": (x_est, x2_est),
        "rs": (r, r2),
        "aa": aa,
        "W": net.W,
        "cov_noise": net.cov_noise
    }

    print("a", aa[0])
    print("c", x2_est[:,15] - x_est[:,15])

    pickle_dump("data/dc_random_noise_norm1", td)
    return td

def exp_random_noise_norm1_p2():
    """
    Injects random noise.
    """
    net = PowerGrid(14)
    td = pickle_load("data/dc_random_noise_norm1")
    net.W = td["W"]
    net.cov_noise = td["cov_noise"]
    z3 = td["zs"][1]

    # Introduce random noise
    z3[11, 15] += 0.05
    z3[19, 15] += 0.1

    x3_est = net.estimate_state(z3)
    r3 = net.calculate_normalized_residuals(z3, x3_est)

    td = {
        "zs": (*td["zs"], z3),
        "x_ests": (*td["x_ests"], x3_est),
        "rs": (*td["rs"], r3),
        "aa": td["aa"]
    }

    print(td["aa"][0])

    print(np.max(r3[:,15]))

    pickle_dump("data/dc_random_noise_norm1_p2", td)
    return td

def exp_random_noise_incomplete_knowledge(model, label, measurements, param_name, 
                                          Ts, inject_start, duration):
    net = PowerGrid(14)
    z = net.create_measurements(measurements, 1)

    param_range = (1, 50)

    config = {
        "H": net.H,
        "Z": z,
        "state_noise": net.state_noise_factor,
        "scenario": "1a"
    }

    Z = config["Z"].copy()
    zs = [z]
    aaa = []
    params = []
    for T in Ts:
        aaa.append([])
        params.append([])
        config["T"] = T
        for ts in range(inject_start, inject_start+duration):
            config["Z"] = Z[:ts]
            config["t"] = ts
            param = uniform(*param_range)
            config[param_name] = param
            aaa[-1].append(model(**config))
            params[-1].append(param)
    
        zs.append(np.copy(z))
        for ts in range(len(aaa[-1])):
            zs[-1][:, inject_start+ts] += aaa[-1][ts].transpose()

    x_ests = []
    for z in zs:
        x_ests.append(net.estimate_state(z))

    rs = []
    for i in range(len(zs)):
        rs.append(net.calculate_normalized_residuals(zs[i], x_ests[i]))

    for r in rs:
        r[r == np.inf] = 0
        r[np.isnan(r)] = 0

    td = {
        "zs": zs,
        "x_ests": x_ests,
        "rs": rs,
        "Ts": Ts,
        "aaa": aaa,
        param_name + "s": params,
        param_name + "_range": param_range
    }

    pickle_dump("data/dc_random_noise_"+label, td)
    return td

def exp_random_noise_random_matrix():
    return exp_random_noise_incomplete_knowledge(
        AnomalyModels.random_matrix, "random_matrix", 2100, "tau", 
        [350, 500, 1000], 1001, 1000
    )

def exp_random_noise_modal_decomposition():
    return exp_random_noise_incomplete_knowledge(
        AnomalyModels.modal_decomposition, "modal_decomposition", 100, "sigma", 
        [20], 31, 30
    )
    
def exp_computational_complexity_complete_knowledge(model, label, 
        nw_size=14, timeout=3600, cfg={}):
    net = PowerGrid(nw_size, double_measurements="double_measurements" in cfg)
    config = {
        "H": net.H,
        "fixed": {},
        "silence": True
    }

    k_range = (0, net.network._ppc['branch'].shape[0] - 1)
    alpha_range = (-0.1, 0.1)
    iterations = 100

    @timeit
    def f(**cfg):
        return model(**cfg)

    outcomes = [] # (k, alpha, elapsed time)
    max_time = timeout # Avoid further iterations if this is exceeded
    current_time = 0
    for i in range(iterations):
        if not i % 10:
            print(i)
        if current_time > max_time:
            print(f"Timed out after {i} iterations")
            break
        k = randint(*k_range)
        alpha = uniform(*alpha_range)
        config["fixed"] = {k: alpha}
        _a, elapsed =  f(**config)
        outcome = (k, alpha, elapsed, _a)
        if np.max(np.abs(_a)) == 0:
            print(label, outcome)
        
        outcomes.append(outcome)
        current_time += elapsed

    td = {
        "label": label, 
        "k_range": k_range,
        "alpha_range": alpha_range,
        "outcomes": outcomes,
    }

    pickle_dump("data/dc_computational_complexity_"+label, td)
    return td

def exp_computational_complexity_norm1(label="norm_1", nw_size=14, cfg={}):
    return exp_computational_complexity_complete_knowledge(
        AnomalyModels.least_effort_norm_1, label, nw_size, cfg=cfg
    )

def exp_computational_complexity_small_ubiquitous(label="small_ubiquitous", nw_size=14, cfg={}):
    return exp_computational_complexity_complete_knowledge(
        AnomalyModels.small_ubiquitous, label, nw_size, cfg=cfg
    )

def exp_computational_complexity_targeted_general(model, label, 
        nw_size=14, timeout=3600, cfg={}):
    net = PowerGrid(nw_size, double_measurements="double_measurements" in cfg)

    config = {
        "H": net.H,
        "fixed": {},
        "silence": True
    }

    line_range = (0, net.network._ppc['branch'].shape[0] - 1)
    alpha_range = (-0.1, 0.1)
    iterations = 100

    @timeit
    def f(**cfg):
        return model(**cfg)

    outcomes = [] # (k, alpha, elapsed time)
    max_time = timeout # Avoid further iterations if this is exceeded
    current_time = 0
    for i in range(iterations):
        if not i % 10:
            print(i)
        if current_time > max_time:
            print(f"Timed out after {i} iterations")
            break
        config["fixed"] = {}
        line = randint(*line_range)
        alpha = uniform(*alpha_range)

        lines = net.network.line.shape[0]
        if line < lines:
            i1 = net.network.line.from_bus[line]
            i2 = net.network.line.to_bus[line]
            if i1 == 0:
               i1, i2 = i2, i1
        if line >= lines: # remainder are in network.trafo
            i1 = net.network.line.from_bus[line - lines]
            i2 = net.network.line.to_bus[line - lines]

        config["fixed"][i1] = alpha
        config["fixed"][i2] = 0

        _a, elapsed =  f(**config)
        outcome = (line, alpha, elapsed, _a)
        if np.max(np.abs(_a)) == 0:
            print(label, outcome)
        outcomes.append(outcome)
        current_time += elapsed

    td = {
        "label": label, 
        "line_range": line_range,
        "alpha_range": alpha_range,
        "outcomes": outcomes,
    }

    pickle_dump("data/dc_computational_complexity_"+label, td)
    return td

def exp_computational_complexity_targeted_norm1(label="targeted_norm1", nw_size=14, cfg={}):
    return exp_computational_complexity_targeted_general(
        AnomalyModels.targeted_least_effort_norm_1, label, nw_size, cfg=cfg
    )

def exp_computational_complexity_targeted_small_ubiquitous(label="targeted_small_ubiquitous", nw_size=14, cfg={}):
    return exp_computational_complexity_targeted_general(
        AnomalyModels.targeted_small_ubiquitous, label, nw_size, cfg=cfg
    )

def exp_computational_complexity_targeted_matching_pursuit(label="targeted_matching_pursuit", nw_size=14, cfg={}):
    return exp_computational_complexity_targeted_general(
        AnomalyModels.targeted_matching_pursuit, label, nw_size, cfg=cfg
    )

def exp_computational_complexity_incomplete_knowledge(model, label, param_name, 
        nw_size=14, timeout=3600, cfg={}):
    net = PowerGrid(nw_size, double_measurements="double_measurements" in cfg)

    T = round(17.5 * net.network._ppc['branch'].shape[0])
    inject_start = T + 1
    duration = 100

    z = net.create_measurements(T + duration + 50, 1)
    param_range = (1, 50)

    config = {
        "H": net.H,
        "T": T,
        "Z": z,
        "state_noise": net.state_noise_factor,
        "scenario": "2"
    }

    @timeit
    def f(**cfg):
        return model(**cfg)
        
    outcomes = []
    max_time = timeout # Avoid further iterations if this is exceeded
    current_time = 0
    Z = config["Z"].copy()
    for ts in range(inject_start, inject_start+duration):
        if current_time > max_time:
            print(f"Timed out after {ts-inject_start} iterations")
            break
        config["Z"] = Z[:ts]
        config["t"] = ts
        param = uniform(*param_range)
        config[param_name] = param
        _a, elapsed = f(**config)
        outcome = (None, None, elapsed, _a) # (k, alpha, elapsed, a)
        outcomes.append(outcome)
        current_time += elapsed

    td = {
        "label": label, 
        "T": T,
        param_name: param_range,
        "outcomes": outcomes,
    }

    pickle_dump("data/dc_computational_complexity_"+label, td)
    return td

def exp_computational_complexity_random_matrix(label="random_matrix", nw_size=14, cfg={}):
    return exp_computational_complexity_incomplete_knowledge(
        AnomalyModels.random_matrix, label, "tau", nw_size, cfg=cfg
    )
    
def exp_computational_complexity_modal_decomposition(label="modal_decomposition", nw_size=14, cfg={}):
    return exp_computational_complexity_incomplete_knowledge(
        AnomalyModels.modal_decomposition, label, "sigma", nw_size, cfg=cfg
    )

def exp_staleness_incomplete_knowledge(model, label, param_name):
    net = PowerGrid(14)
    z = net.create_measurements(700, 1)
    z2 = np.copy(z)

    config = {
        "H": net.H,
        "T": 500,
        "Z": z,
        "state_noise": net.state_noise_factor,
        "scenario": 2,
        param_name: 50,
    }

    inject_start = 501
    duration = 100
    
    config["Z"] = config["Z"][:inject_start]
    config["t"] = inject_start
    aa = [model(**config)] * duration

    for ts in range(len(aa)):
        z2[:, inject_start+ts] += aa[ts].transpose()

    x_est = net.estimate_state(z)
    x2_est = net.estimate_state(z2)

    r = net.calculate_normalized_residuals(z, x_est)
    r2 = net.calculate_normalized_residuals(z2, x2_est)

    r[r == np.inf] = 0
    r2[r2 == np.inf] = 0
    r[np.isnan(r)] = 0
    r2[np.isnan(r2)] = 0

    td = {
        "zs": (z, z2),
        "x_ests": (x_est, x2_est),
        "rs": (r, r2),
        "aa": aa,
        param_name: config[param_name]
    }

    pickle_dump(f"data/dc_staleness_"+label, td)
    return td

def exp_staleness_random_matrix():
    return exp_staleness_incomplete_knowledge(
        AnomalyModels.random_matrix, "random_matrix", "tau"
    )

def exp_staleness_modal_decomposition():
    return exp_staleness_incomplete_knowledge(
        AnomalyModels.modal_decomposition, "modal_decomposition", "sigma"
    )

def exp_ac_injection_in_dc():
    net = PowerGrid(14)
    z = net.create_measurements(1, 1)
    z2 = np.copy(z)

    target_sensor = 1
    alpha = 0.1

    config = {
        "H": net.H,
        "fixed": {1 : 0.1},
    }

    a = pickle_load("data/ac_dc_norm_1_comparison")["a"]

    z2[:, 0] += a[:20].transpose()
    
    x_est = net.estimate_state(z)
    x2_est = net.estimate_state(z2)

    r = net.calculate_normalized_residuals(z, x_est)
    r2 = net.calculate_normalized_residuals(z2, x2_est)

    r[r == np.inf] = 0
    r2[r2 == np.inf] = 0

    td = {
        "zs": (z, z2),
        "x_ests": (x_est, x2_est),
        "rs": (r, r2),
    }

    print(np.nanmax(np.abs(r2[:,0] - r[:,0])))

    pickle_dump("data/ac_injection_in_dc", td)
    return td

def exp_computational_complexity_n_bus():
    tests = {
        #"norm1": exp_computational_complexity_norm1,
        #"small_ubiquitous": exp_computational_complexity_small_ubiquitous,
        #"targeted_norm1": exp_computational_complexity_targeted_norm1,
        #"targeted_small_ubiquitous": exp_computational_complexity_targeted_small_ubiquitous,
        #"targeted_matching_pursuit": exp_computational_complexity_targeted_matching_pursuit,
        "random_matrix": exp_computational_complexity_random_matrix,

        # This takes a lot of time, so should only be used with smaller grids (<= 14)
        #"modal_decomposition": exp_computational_complexity_modal_decomposition
    }

    tds = []

    nw_sizes = [14, 39, 118, 200, 1354]
    for nw_size in nw_sizes:
        td = {}
        for key, f in tests.items():
            print(f"Running: {key} ({nw_size}-bus)")
            t1 = time()
            td[key] = f(key+f"_{nw_size}", nw_size)
            t2 = time()
            print("Elapsed (total): ", t2 - t1)

        pickle_dump(f"data/dc_computational_complexity_{nw_size}_summary_rmt", td)
        tds.append(td)

    return tds

def exp_computational_complexity_double_measurements():
    tests = {
        "norm1": exp_computational_complexity_norm1,
        "small_ubiquitous": exp_computational_complexity_small_ubiquitous,
        "targeted_norm1": exp_computational_complexity_targeted_norm1,
        "targeted_small_ubiquitous": exp_computational_complexity_targeted_small_ubiquitous,
        "targeted_matching_pursuit": exp_computational_complexity_targeted_matching_pursuit,
        "random_matrix": exp_computational_complexity_random_matrix,

        # This takes a lot of time, so should only be used with smaller grids (<= 14)
        #"modal_decomposition": exp_computational_complexity_modal_decomposition
    }

    tds = []

    nw_sizes = [14, 39, 118]
    for nw_size in nw_sizes:
        td = {}
        for test, f in tests.items():
            print(f"Running: {test} (ieee-{nw_size})")
            t1 = time()
            td[test] = f(test+f"_double_{nw_size}", nw_size, {"double_measurements":True})
            t2 = time()
            print("Elapsed (total): ", t2 - t1)

        pickle_dump(f"data/dc_computational_complexity_double_{nw_size}_summary_rmt", td)
        tds.append(td)

    return tds

def targeted_get_fixed(net, k, alpha):
        lines = net.network.line.shape[0]
        if k < lines:
            i1 = net.network.line.from_bus[k]
            i2 = net.network.line.to_bus[k]
            if i1 == 0:
               i1, i2 = i2, i1
        if k >= lines: # remainder are in network.trafo
            i1 = net.network.line.from_bus[k - lines]
            i2 = net.network.line.to_bus[k - lines]

        return {i1: alpha, i2: 0}

def exp_unfolding_risky_sensors_gather(threshold=1e-5):
    iterations = 1000
    tests = [
        "norm1",
        "small_ubiquitous",
        "targeted_norm1",
        "targeted_small_ubiquitous",
        "targeted_matching_pursuit",
        "random_matrix",
    ]

    fs = {
        "norm1": AnomalyModels.least_effort_norm_1,
        "small_ubiquitous": AnomalyModels.small_ubiquitous,
        "targeted_norm1": AnomalyModels.targeted_least_effort_norm_1,
        "targeted_small_ubiquitous": AnomalyModels.targeted_small_ubiquitous,
        "targeted_matching_pursuit": AnomalyModels.targeted_matching_pursuit,
        "random_matrix": AnomalyModels.random_matrix
    }

    net = PowerGrid(39)

    k_range = (0, net.network._ppc['branch'].shape[0] - 1)
    alpha_range = (-0.1, 0.1)
    tau_range = (0.001, 0.01)
    T = round(20 * net.network._ppc['branch'].shape[0])

    sensor_count = np.zeros(k_range[1] + 1) # {z_index : occurrence in a vector}
    sensor_counts = {} # {test_id : [sensor occurences]}
    a_vectors = {} # {test_id : [a0, a1, ...]}
    k_counts = np.copy(sensor_count)

    config = {
        "H": net.H,
        "T": T,
        "t": T+1,
        "silence": True,
        "state_noise": net.state_noise_factor,
        "scenario": "1b" # random_matrix: choose largest eigenvalues
    }
    
    for _, test in enumerate(tests):
        print("Testing:", test)
        counts = np.zeros(k_range[1] + 1)
        a_vs = []
        for j in range(iterations):
            if not j % (iterations // 10):
                print(j)
            if test == "random_matrix":
                # This change ensures that the a vector returned by random_matrix
                # does not contain NaN values, with the side effect of significantly
                # increasing the values returned by the eigenvalue decomposition
                # (thus necessitating a lower tau range).
                net.state_noise_factor = 1e-2
                net.measurement_noise_factor = 1e-3
                config["tau"] = uniform(*tau_range)
                config["Z"] = net.create_measurements(config["t"], 1)
            else:
                # Default noise settings
                net.state_noise_factor = 1e-3
                net.measurement_noise_factor = 1e-2
                k = randint(*k_range)
                alpha = uniform(*alpha_range)
                if "targeted" in test:
                    config["fixed"] = targeted_get_fixed(net, k, alpha)
                else:
                    config["fixed"] = {k: alpha}

                for k, alpha in config["fixed"].items():
                    if alpha:
                        k_counts[k] += 1

            a = fs[test](**config)
            a_vs.append(np.copy(a))
            a[a < threshold] = 0
            a[a >= threshold] = 1
            a[np.isnan(a)] = 0
            counts += a

        sensor_counts[test] = counts
        sensor_count += counts
        a_vectors[test] = a_vs

    # [(index, count), ...] in descending order
    sensor_count_sorted = sorted([(i, c) for (i, c) in enumerate(sensor_count)],
                                 key = lambda t: t[1],
                                 reverse = True)

    td = {
        "a_vectors": a_vectors,
        "k_counts": k_counts,
        "sensor_counts": sensor_counts,
        "sensor_count": sensor_count,
        "sensor_count_sorted": sensor_count_sorted,
        "nw_size": 39,
        "k_range": k_range,
        "T": T,
        "tau_range": tau_range
    }

    print(*sensor_count_sorted[:10], sep="\n")

    pickle_dump(f"data/dc_unfolding_risky_sensors", td)
    return td

def exp_unfolding_risky_sensors_run():
    td = pickle_load(f"data/dc_unfolding_risky_sensors_gather")
    net = PowerGrid(td["nw_size"])
    
    avs = td["a_vectors"]
    sensor_counts = td["sensor_count_sorted"] # Sorted in descending order
    secured = 1 # Secure the n most utilized sensors from the gathering stage

    # One time step
    zs = { # Without any injections
        "norm1": [],
        "small_ubiquitous": [],
        "targeted_norm1": [],
    }
    fs = {
        "norm1": AnomalyModels.least_effort_norm_1,
        "small_ubiquitous": AnomalyModels.small_ubiquitous,
        "targeted_norm1": AnomalyModels.targeted_least_effort_norm_1
    }


    z2s = zs.copy() # With the injection from the gathering stage
    z3s = zs.copy() # With new injection where one sensor is secured

    x_ests = zs.copy()
    x2_ests = zs.copy()
    x3_ests = zs.copy()

    rs = zs.copy()
    r2s = zs.copy()
    r3s = zs.copy()

    secured_sensors = [o[0] for o in sensor_counts[:secured]]

    k_range = td["k_range"]
    alpha_range = (-0.1, 0.1)
    threshold = 1e-5

    cfg = {
        "H": net.H,
        "secure": secured_sensors,
        "silence": True
    }

    sensor_count = np.zeros(k_range[1] + 1) # {z_index : occurrence in a vector}
    sensor_counts = {} # {test_id : [sensor occurences]}
    a_vectors = {} # {test_id : [a0, a1, ...]}
    k_counts = np.copy(sensor_count)

    for test in avs:
        print(test)
        if test not in fs:
            continue
        counts = np.zeros(k_range[1] + 1) # For this test
        a_vectors[test] = []
        
        zs[test] = net.create_measurements(len(avs[test]), 1)
        z2s[test] = zs[test].copy()
        z3s[test] = zs[test].copy()
        for i, a in enumerate(avs[test]):
            z2s[test][:, i] += a.transpose()
            
            k = randint(*k_range)
            alpha = uniform(*alpha_range)
            if "targeted" in test:
                cfg["fixed"] = targeted_get_fixed(net, k, alpha)
            else:
                cfg["fixed"] = {k: alpha}

            # Create a new injection vector based on the new secured sensor
            b = fs[test](**cfg)

            a_vectors[test].append(np.copy(b))
            b[b < threshold] = 0
            b[b >= threshold] = 1
            b[np.isnan(b)] = 0
            counts += b

            z3s[test][:, i] += b.transpose()

        sensor_counts[test] = counts        

        # For reference, also calculate residuals
        x_ests[test] = net.estimate_state(zs[test])
        x2_ests[test] = net.estimate_state(z2s[test])
        x3_ests[test] = net.estimate_state(z3s[test]) 

        rs[test] = net.calculate_normalized_residuals(zs[test], x_ests[test])
        r2s[test] = net.calculate_normalized_residuals(z2s[test], x2_ests[test])
        r3s[test] = net.calculate_normalized_residuals(z3s[test], x3_ests[test])
    
    td = {
        "zs": zs,
        "z2s": z2s,
        "z3s": z3s,
        "x_ests": x_ests,
        "x2_ests": x2_ests,
        "x3_ests": x3_ests,
        "rs": rs, 
        "r2s": r2s, 
        "r3s": r3s,
        "secured_sensors": secured_sensors,
        "sensor_counts" : sensor_counts
    }

    pickle_dump("data/dc_unfolding_risky_sensors_run", td)
    return td

def exp_multiplicity():
    net = PowerGrid(39)

    config = {
        "H": net.H,
        "fixed": {0: -0.1},
        "multiple_solutions": 3,
    }

    @timeit
    def f(**c):
        return AnomalyModels.least_effort_big_m(**c)

    sols, elapsed = f(**config)
    print(sols, elapsed)
    return sols, elapsed

def exp_norm1_vs_big_m():
    net = PowerGrid(39)

    config = {
        "H": net.H,
        "fixed": {},
        "M": 1000,
        "silence": True
    }

    alpha = -0.1
    k_range = (0, 46)
    threshold = 1e-5

    norm_1_as = []
    big_m_as = []
    outcomes = [] # (norm_1_nonzeros, big_m_nonzeros)
    for k in range(*k_range):
        config["fixed"] = {k: alpha}

        _as = []
        _as.append(AnomalyModels.least_effort_norm_1(**config))
        _as.append(AnomalyModels.least_effort_big_m(**config))

        norm_1_as.append(_as[0].copy())
        big_m_as.append(_as[1].copy())

        for i, a in enumerate(_as):
            a[abs(a) < threshold] = 0
            a[abs(a) >= threshold] = 1
        t = tuple(sum(a) for a in _as)
        outcomes.append(t)

    td = {
        "alpha": alpha,
        "norm_1_as": norm_1_as,
        "big_m_as": big_m_as,
        "outcomes": outcomes,
    }

    print(*[a for a in outcomes if a[0] > a[1]], sep="\n")

    pickle_dump("data/dc_norm1_vs_big_m", td)
    return td

def exp_norm1_leverage_single():
    net = PowerGrid(14)
    config = {
        "H": net.H,
        "fixed": {18: 0.1}
    }
    a = AnomalyModels.least_effort_norm_1(**config)

    # Should only contain 1 element
    print(a)
    return(a)

def exp_norm1_leverage_double():
    net = PowerGrid(14, double_measurements=True)
    config = {
        "H": net.H,
        "fixed": {36: 0.1}
    }
    a = AnomalyModels.least_effort_norm_1(**config)
    
    # Should only contain 2 elements next to each other
    # (0.1 and -0.1)
    print(a)
    return(a)


if __name__ == "__main__":
    np.set_printoptions(edgeitems=10, linewidth=180)

    print()




