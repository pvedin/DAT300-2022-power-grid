from powergrid import *

def println(*args, **kwargs):
    if "sep" in kwargs:
        del kwargs["sep"]
    print(*args, **kwargs, sep="\n")

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
    anomalies = net.check_for_anomalies()
    println("Anomalies:", [(i, r[i[0], i[1]]) for i in anomalies])
    print(len(anomalies))


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
    anomalies = net.check_for_anomalies()
    println("Anomalies:", [(i, r[i[0], i[1]]) for i in anomalies])
    print(len(anomalies))

def test_norm_1():
    print("Testing: least_effort_norm_1")
    net = PowerGrid(14)
    z = net.create_measurements(100, 1)

    config = {
        "H": net.H,
        "fixed": {1:30},
        "a_bounds": (-1000, 1000),
        "c_bounds": (-1000, 1000),
    }
    a = AnomalyModels.least_effort_norm_1(**config)

    println("a", a)
    z[:,5] += a

    a = np.zeros(z[:, 1].shape)
    a[1] = abs(z[1, 6]) * 0.5
    z[:, 6] += a

    x_est = net.estimate_state(z)
    println("x_est", x_est)
    r = net.calculate_normalized_residuals()
    
    println("Normalized residuals:", r, r.shape)
    anomalies = net.check_for_anomalies()
    println("Anomalies:", [(i, r[i[0], i[1]]) for i in anomalies])
    print(len(anomalies))

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
    anomalies = net.check_for_anomalies()
    println("Anomalies:", [(i, r[i[0], i[1]]) for i in anomalies])
    print(len(anomalies))

def test_matching_pursuit():
    print("Testing: Matching pursuit")
    net = PowerGrid(14)
    z = net.create_measurements(1, 1)
    z = z.repeat(3, axis=1)

    config = {
        "H": net.H,
        "fixed": {1:10},
    }
    possible_as = AnomalyModels.targeted_matching_pursuit(**config)
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
    anomalies = net.check_for_anomalies()
    println("Anomalies:", [(i, r[i[0], i[1]]) for i in anomalies])
    print(len(anomalies))

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
    anomalies = net.check_for_anomalies()
    println("Anomalies:", [(i, r[i[0], i[1]]) for i in anomalies])
    print(len(anomalies))

def test_random_matrix():
    print("Testing: random matrix")
    net = PowerGrid(14)
    z = net.create_measurements(1000, 1)
    print(z.shape)
    config = {
        "H": net.H,
        "Z": z,
        "t": 750,
        "fixed": {2:10}
    }

    a = AnomalyModels.random_matrix(**config)
    println(a)
    println(type(z[:,750]), z[:,750].shape, z[:,750])
    z[:,750] += a
    x_est = net.estimate_state(z)
    println(x_est[:,749:752])

def test_for_anomalies():
    print("Testing: how much can a single measurement be changed before "
          + "the measurement is considered anomalous?")
    net = PowerGrid(14)
    z = net.create_measurements(20, 1)

    # Affect the fifth element in the following measurements by the given
    # percentages (e.g. for time step 6 the value of the element will be
    # 1 + 2 = 3x the regular value)
    tss =                  [  2,   3,   4, 5, 6, 7,  8,  9, 10]
    percentage_increases = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
    for i in range(len(tss)):
        a = np.zeros(z[:, tss[i]].shape)
        a[4] = abs(z[1, tss[i]]) * percentage_increases[i]
        z[:, tss[i]] += a

    x_est = net.estimate_state(z)
    r = net.calculate_normalized_residuals()
    
    println("Normalized residuals:", r, r.shape)
    anomalies = net.check_for_anomalies()
    println("Anomalies:", [(i, r[i[0], i[1]]) for i in anomalies])
    print(len(anomalies))

if __name__ == "__main__":
    np.set_printoptions(edgeitems=10, linewidth=180)

    test_targeted_least_effort()