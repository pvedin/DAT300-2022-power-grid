from powergrid import *
from itertools import chain, combinations
from random import uniform, randint, sample

def println(*args, **kwargs):
    if "sep" in kwargs:
        del kwargs["sep"]
    print(*args, **kwargs, sep="\n")

def print_anomalies(net, r):
    anomalies = net.check_for_anomalies(r)
    # (row, col) == (measurement_index, timestamp)
    println("Anomalies:", [(f"ts {i[0]} index {i[1]}", r[i[0], i[1]]) for i in anomalies])
    print(len(anomalies))

def test_blueprint():
    print("Example test")
    net = PowerGrid(14) # Alternative arguments: "IEEE-14", "14"
    z = net.create_measurements(100, 1) # 'x' measurements using data strategy 'y'
    config = {
        "H": net.H,
        "fixed": {}, # {k:delta},
        "a_bounds": (-1000, 1000), # (lower, upper)
        "c_bounds": (-1000, 1000), # (lower, upper)
    }
    config["fixed"][1] = 5 # Change the second element by +5
    a = AnomalyModels.least_effort_norm_1(**config)

    for time_step in range(10, 21):
        z[:, time_step] += a.transpose()

    x_est = net.estimate_state(z) # Arguments can be provided to override stored values
    r = net.calculate_normalized_residuals()
    print_anomalies(net, r)


def test_demo():
    print("Presentation demo")
    net = PowerGrid(14) # Alternative arguments: "IEEE-14", "14"
    z = net.create_measurements(1, 1) # 'x' measurements using data strategy 'y'
    z = z.repeat(3, axis=1) # Keep three versions of the same array
    println(f"Created measurements {z.shape} with strategy 1:", z)

    # Introduce anomalies to the second using the least-effort model
    config = {
        "H": net.H,
        "z": None,
        "fixed": {},
        "a_bounds": (-1000, 1000),
        "c_bounds": (-1000, 1000),
    }
    config["fixed"][1] = abs(z[1, 1]) * 0.2
    a = AnomalyModels.least_effort_norm_1(**config)
    println("Attack vector:", a.transpose())
    z[:, 1] += a.transpose()
    
    # Naively try to tamper with the last time step
    a = np.zeros(z[:, 1].shape)
    a[1] = abs(z[1, 2]) * 0.2
    z[:, 2] += a
    println(z)

    x_est = net.estimate_state(z) # Arguments can be provided to override stored values
    println("Estimated state:", x_est, x_est.shape)
    r = net.calculate_normalized_residuals()
    println("Normalized residuals:", r, r.shape)
    print_anomalies(net, r)

def test_norm_1():
    print("Testing: least_effort_norm_1")
    net = PowerGrid(14)
    z = net.create_measurements(100, 1)

    config = {
        "H": net.H,
        "fixed": {2:5},
        "a_bounds": (-1000, 1000),
        "c_bounds": (-1000, 1000),
    }
    a = AnomalyModels.least_effort_norm_1(**config)

    println("a", a)
    z[:,5] += a

    a = np.zeros(z[:, 1].shape)
    #a[1] = abs(z[2, 6]) * 0.1
    z[2, 6] += 0.1

    x_est = net.estimate_state(z)
    println("x_est", x_est)
    r = net.calculate_normalized_residuals()
    
    println("Normalized residuals:", r, r.shape)
    print_anomalies(net, r)

def test_big_m():
    print("Testing: least effort (big-M)")
    net = PowerGrid(14)
    z = net.create_measurements(1, 1)
    z = z.repeat(3, axis=1)
    config = {
        "H": net.H,
        "z": None,
        "fixed": {1: 5},
        "a_bounds": (-1000, 1000),
        "c_bounds": (-1000, 1000),
        "M": 1000
    }
    a1 = AnomalyModels.targeted_least_effort_big_m(**config)
    z[:, 1] += a1
    println("z", z)
    x_est = net.estimate_state(z)
    println("x_est:", x_est, x_est.shape)
    r = net.calculate_normalized_residuals()
    print_anomalies(net, r)

def test_targeted_norm_1():
    print("Testing: least_effort_norm_1")
    net = PowerGrid(14)
    z = net.create_measurements(100, 1)

    config = {
        "H": net.H,
        "fixed": {1:-10},
        "a_bounds": (-1000, 1000),
        "c_bounds": (-1000, 1000),
    }
    a = AnomalyModels.targeted_least_effort_norm_1(**config)

    println("a", a)
    z[:,5] += a

    a = np.zeros(z[:, 1].shape)
    a[1] = abs(z[1, 6]) * 0.5
    z[:, 6] += a

    x_est = net.estimate_state(z)
    println("x_est", x_est)
    r = net.calculate_normalized_residuals()
    
    println("Normalized residuals:", r, r.shape)
    print_anomalies(net, r)

def test_targeted_least_effort():
    print("Testing: Targeted least effort (norm-1 and big-M)")
    net = PowerGrid(14)
    z = net.create_measurements(1, 1)
    z = z.repeat(3, axis=1)
    config = {
        "H": net.H,
        "z": None,
        "fixed": {1: 5},
        "a_bounds": (-1000, 1000),
        "c_bounds": (-1000, 1000),
        "M": 1000
    }
    a1 = AnomalyModels.targeted_least_effort_norm_1(**config)
    a2 = AnomalyModels.targeted_least_effort_big_m(**config)
    z[:, 1] += a1
    z[:, 2] += a2
    println("z", z)
    x_est = net.estimate_state(z)
    println("x_est:", x_est, x_est.shape)
    r = net.calculate_normalized_residuals()
    print_anomalies(net, r)

def test_matching_pursuit():
    print("Testing: Matching pursuit")
    net = PowerGrid(14)
    z = net.create_measurements(1, 1)
    z = z.repeat(3, axis=1)
    print("shape",z.shape)

    config = {
        "H": net.H,
        "fixed": {1:-10},
    }
    possible_as = AnomalyModels.targeted_matching_pursuit(**config)
    if not possible_as:
        print("Infeasible!")
        exit()
    println("Possible as:", possible_as)
    z[:, 1] += possible_as[0][1].transpose()

    config = {
        "H": net.H,
        "fixed": {1:100},
    }
    possible_as = AnomalyModels.targeted_matching_pursuit(**config)
    println("Possible as:", possible_as)
    z[:, 2] += possible_as[0][1].transpose()

    x_est = net.estimate_state(z)
    println("x_est:", x_est)
    r = net.calculate_normalized_residuals()
    print_anomalies(net, r)

def test_modal_decomposition():
    print("Testing: modal decomposition")
    net = PowerGrid(14)
    z = net.create_measurements(2, 1)
    z = z.repeat(2, axis=1)
    try:
        z_a1 = AnomalyModels.modal_decomposition(z[:, 2])
        print("ts 2 ok!")
        z_a2 = AnomalyModels.modal_decomposition(z[:, 3])
        print("ts 3 ok!")
    except AssertionError:
        println("Infeasible!", z)
        exit()
        
    z[:, 2] = z_a1
    z[:, 3] = z_a2
    println("z:", z)
    x_est = net.estimate_state(z)
    println("x_est:", x_est)
    r = net.calculate_normalized_residuals()
    print_anomalies(net, r)

def test_random_matrix():
    print("Testing: random matrix")
    net = PowerGrid(14)
    z = net.create_measurements(1000, 1)
    print(z.shape)
    config = {
        "H": net.H,
        "Z": z,
        "t": 750,
        "fixed": {2:-100}
    }

    a = AnomalyModels.random_matrix(**config)
    println(a)
    println(type(z[:,750]), z[:,750].shape, z[:,750])
    z[:,750] += a
    x_est = net.estimate_state(z)
    println("Estimated states:", x_est[:,749:752])
    r = net.calculate_normalized_residuals()
    print("Residuals:", r[:, 749:752])
    print_anomalies(net, r)

def test_for_anomalies():
    print("Testing: how much can a single measurement be changed before "
          + "the measurement is considered anomalous?")
    net = PowerGrid(14)
    z = net.create_measurements(1, 1, env_noise = False)
    z = z.repeat(20, axis=1)

    print("z17 before", z[:,17])

    # Affect the fifth element in the following measurements by the given
    # percentages (e.g. for time step 6 the value of the element will be
    # 1 + 2 = 3x the regular value)
    tss =                  [  2,   3,   4, 5, 6, 7,  8,  9, 10]
    percentage_increases = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    for i in range(len(tss)):
        a = np.zeros(z[:, tss[i]].shape)
        a[1] = abs(z[1, tss[i]]) * percentage_increases[i]
        z[:, tss[i]] += a
        #println(tss[i], z[:, tss[i]], a)

    x_est = net.estimate_state(z)
    r = net.calculate_normalized_residuals()

    print("z17 after", z[:,17])
    
    println("Normalized residuals:", r, r.shape)
    anomalies = net.check_for_anomalies()
    pi = percentage_increases
    println("Anomalies:", *[(i, pi[tss.index(i[0])] if i[0] in tss else 0, 
                             r[i[0], i[1]]) for i in anomalies])
    print(len(anomalies))

def test_sensor_count():
    # Attempt to see which sensors are more commonly used in least_effort_norm_1
    # injections
    net = PowerGrid(14)
    aa = []
    config = {
        "H": net.H,
        "fixed": {}, 
        "a_bounds": (-1000, 1000),
        "c_bounds": (-1000, 1000),
        "silence": True
    }
    # This also affects what other values need to be modified
    delta = -0.1
    
    # Keep track of how many injections are considered feasible
    total = 0

    for k_values in range(1,4):
        # Exhaustively try combinations of at most three target values
        kss = tuple(combinations(range(net.H.shape[0]), k_values))
        for ks in kss:
            total += 1
            config["fixed"] = {}
            for k in ks: 
                config["fixed"][k] = delta
            a = AnomalyModels.least_effort_norm_1(**config)
            if np.count_nonzero(a):
                aa.append(a.tolist())

    sensor_count = {i:0 for i in range(20)}
    for a in aa:
        for i, v in enumerate(a):
            if abs(v) > 1e-14:
                sensor_count[i] += 1

    results = sorted(list(sensor_count.items()), key=lambda t: t[1], reverse=True)

    print(f"Out of {total} combinations, {len(aa)} were feasible:")
    print("(index, count)", "% frequency")
    for res in results:
        print(res, round(res[1] / len(aa), 3))

def test_small_ubiquitous():
    print("Testing: small_ubiquitous")
    net = PowerGrid(14)
    z = net.create_measurements(3, 1)

    config = {
        "H": net.H,
        "fixed": {1:1},
        "a_bounds": (-1000, 1000),
        "c_bounds": (-1000, 1000),
    }
    a = AnomalyModels.small_ubiquitous(**config)

    println("a", a)
    print(z[:,2])
    z[:,2] += a
    print(z[:,2])

    x_est = net.estimate_state(z)
    println("x_est", x_est)
    r = net.calculate_normalized_residuals()
    
    println("Normalized residuals:", r, r.shape)
    print_anomalies(net, r)

def test_targeted_small_ubiquitous():
    print("Testing: targeted_small_ubiquitous")
    net = PowerGrid(14)
    z = net.create_measurements(3, 1)

    config = {
        "H": net.H,
        "fixed": {1:2},
        "a_bounds": (-1000, 1000),
        "c_bounds": (-1000, 1000),
    }
    a = AnomalyModels.targeted_small_ubiquitous(**config)

    println("a", a)
    print(z[:,2])
    z[:,2] += a
    print(z[:,2])

    x_est = net.estimate_state(z)
    println("x_est", x_est)
    r = net.calculate_normalized_residuals()
    
    println("Normalized residuals:", r, r.shape)
    print_anomalies(net, r)

if __name__ == "__main__":
    np.set_printoptions(edgeitems=10, linewidth=180)

    test_norm_1()
    #test_big_m()
    #test_targeted_least_effort()
    #test_small_ubiquitous()
    #test_targeted_small_ubiquitous()
    #test_matching_pursuit()
    #test_modal_decomposition()
    #test_random_matrix()
    #test_for_anomalies()
    #test_sensor_count()
    

