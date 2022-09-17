from powergrid import *

def test_demo():
    print("Presentation demo")
    net = PowerGrid(14) # Alternative arguments: "IEEE-14", "14"
    z = net.create_measurements(1, 1) # 'x' measurements using data strategy 'y'
    z = z.repeat(3, axis=1) # Keep three versions of the same array
    z_before = np.copy(z)
    print(f"Created measurements {z.shape} with strategy 1:", z, sep="\n")

    # Introduce anomalies to the last two time steps using the least-effort model
    config = {
        "H": net.H,
        "z": None,
        "k": 1,  # tamper with 2nd measurement
        "delta": None, # alter by an absolute amount
        "a_bounds": None, # (lower, upper)
        "c_bounds": None, # (lower, upper)
        "M": 1000,
    }
    x_est_pure = net.estimate_state(z)
    config["z"] = z[:, 1]
    config["delta"] = 50 #abs(z[config["k"], 1]) * 0.2
    config["a_bounds"] = (-1000, 1000)
    config["c_bounds"] = (-1000, 1000)
    a = AnomalyModels.least_effort_norm_1(**config)
    print("Attack vector:")
    print(a.transpose())
    z[:,1] += a.transpose()
    print(z)
    print(net.H)
    

    # Naively try to tamper with the last time step
    a = np.zeros(z[:, 1].shape)
    a[4] = abs(z[1, 2]) * 0.2
    z[:, 2] += a

    x_est = net.estimate_state(z) # Arguments can be provided to override stored values
    print("Estimated state:", x_est, x_est.shape, sep="\n")
    r = net.calculate_normalized_residuals()
    print("Normalized residuals:", r, r.shape, sep="\n")
    anomalies = net.check_for_anomalies()
    print("Anomalies:", [(i, r[i[0], i[1]]) for i in anomalies])
    print(len(anomalies))

def test_random_matrix():
    print("Testing: case 14")
    net = PowerGrid(14) # Alternative arguments: "IEEE-14", "14"
    z = net.create_measurements(1000, 1) # 'x' measurements using data strategy 'y'
    print(z.shape)
    config = {
        "H": net.H,
        "Z": z,
        "t": 750,
        "k": 2,
        "delta": 10
    }

    a = AnomalyModels.random_matrix(**config)
    print(a)
    print(type(z[:,750]), z[:,750].shape, z[:,750])
    z[:,750] += a
    x_est = net.estimate_state(z)
    print(x_est[:,749:752])

def test_for_anomalies():
    print("Testing: case 14")
    net = PowerGrid(14) # Alternative arguments: "IEEE-14", "14"
    z = net.create_measurements(10, 1) # 'x' measurements using data strategy 'y'
    a = np.zeros(z[:, 1].shape)
    a[4] = abs(z[1, 2]) * 0.5
    z[:, 2] += a

    x_est = net.estimate_state(z)
    r = net.calculate_normalized_residuals()
    
    print("Normalized residuals:", r, r.shape, sep="\n")
    anomalies = net.check_for_anomalies()
    print("Anomalies:", [(i, r[i[0], i[1]]) for i in anomalies])
    print(len(anomalies))

def test_norm_1():
    print("Testing: least_effort_norm_1")
    net = PowerGrid(14) # Alternative arguments: "IEEE-14", "14"
    z = net.create_measurements(100, 1) # 'x' measurements using data strategy 'y'

    config = {
        "H": net.H,
        "k": 1,  # tamper with 2nd measurement
        "delta": 30, # alter by an absolute amount
        "a_bounds": (-1000, 1000), # (lower, upper)
        "c_bounds": (-1000, 1000), # (lower, upper)
    }
    a = AnomalyModels.least_effort_norm_1(**config)

    print("a", a)
    z[:,5] += a

    a = np.zeros(z[:, 1].shape)
    a[1] = abs(z[1, 6]) * 0.5
    z[:, 6] += a

    x_est = net.estimate_state(z)
    print("x_est", x_est)
    r = net.calculate_normalized_residuals()
    
    print("Normalized residuals:", r, r.shape, sep="\n")
    anomalies = net.check_for_anomalies()
    print("Anomalies:", [(i, r[i[0], i[1]]) for i in anomalies])
    print(len(anomalies))

if __name__ == "__main__":
    np.set_printoptions(edgeitems=10, linewidth=180)

    test_norm_1()