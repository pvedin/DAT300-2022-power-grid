from os.path import isfile
from pickle import dump, load
from powergrid_ac import *


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

def exp_staleness_ac_norm1():
    net = PowerGrid(14)
    z = net.create_measurements(150, env_noise=True)
    z2 = np.copy(z)

    z_x_ests, x_ests, Hs = net.estimate_state(z, restarts=1)

    inject_start = 25
    duration = 100

    config = {
        "net": net,
        "H": Hs[inject_start],
        "x_est": x_ests[inject_start],
        "z_x_est": z_x_ests[inject_start],
        "fixed": {1: 0.01},
        "silence": True
    }

    a, _c = AnomalyModels.least_effort_norm_1(**config)

    for ts in range(inject_start, inject_start + duration):
        z2[:, ts] += a

    Hs2, x2_ests, z_x2_ests = net.estimate_state(z2, restarts=1)

    r = net.calculate_normalized_residuals(z, x_ests, Hs)
    r2 = net.calculate_normalized_residuals(z2, x2_ests, Hs2)

    td = {
        "zs": (z, z2),
        "rs": (r, r2),
        "a": a,
        "fixed": config["fixed"]
    }
    pickle_dump("data/ac_staleness_norm1", td)
    return td

def exp_ac_dc_norm1_comparison():
    net = PowerGrid(14)
    z = net.create_measurements(1, env_noise=True)
    z2 = np.copy(z)

    Hs, x_ests, z_x_ests = net.estimate_state(z, restarts=1)

    config = {
        "net": net,
        "H": Hs[0],
        "x_est": x_ests[0],
        "z_x_est": z_x_ests[0],
        "fixed": {1: 0.1},
        "silence": True,
        "secure": list(range(20, 41))
    }

    a, _c = AnomalyModels.least_effort_norm_1(**config)

    z2[:, 0] += a

    Hs2, x2_ests, _z_x2_ests = net.estimate_state(z2, restarts=1)

    r = net.calculate_normalized_residuals(Hs, x_ests, z)
    r2 = net.calculate_normalized_residuals(Hs2, x2_ests, z2)

    td = {
        "zs": (z, z2),
        "rs": (r, r2),
        "a": a,
        "fixed": config["fixed"],
        "secure": config["secure"]
    }
    pickle_dump("data/ac_dc_norm_1_comparison", td)
    return td

def exp_dc_injection_in_ac():
    net = PowerGrid(14)
    z = net.create_measurements(1, env_noise=True)
    z2 = np.copy(z)

    Hs, x_ests, z_x_ests = net.estimate_state(z, restarts=1)


    a = pickle_load("data/dc_norm1")["aa"][0] # fixed: {1: 0.1}
    a = np.hstack((a, np.zeros(21)))
    
    print(a)

    z2[:, 0] += a

    Hs2, x2_ests, _z_x2_ests = net.estimate_state(z2, restarts=1)

    r = net.calculate_normalized_residuals(Hs, x_ests, z)
    r2 = net.calculate_normalized_residuals(Hs2, x2_ests, z2)

    td = {
        "zs": (z, z2),
        "rs": (r, r2),
        "a": a,
    }
    pickle_dump("data/dc_injection_in_ac", td)
    return td



if __name__ == "__main__":
    np.set_printoptions(edgeitems=10, linewidth=180)

    exp_staleness_ac_norm1()

