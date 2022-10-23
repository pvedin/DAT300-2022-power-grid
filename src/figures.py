from os.path import isfile
from pickle import load
import numpy as np
import matplotlib.pyplot as plt


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

folder = "figures/"
txt_discrete = "Discrete time steps"
txt_residuals = "max(abs({normalized residuals})\n[p.u.]"
txtf_measurement = lambda f,t: "Measurement $P_{"+str(f)+"\\rightarrow"+str(t)+"}$ [p.u.]"
txtf_state = lambda n: "State $\hat{\\theta}_{"+str(n)+"}$ [rad]"

# P_{x->y} for each sensor in the measurement vector for the IEEE-14 bus
ps = [(1,2), (1,5), (2,3), (2,4), (2,5), (3,4), (4,5), # \
      (6,11), (6,12), (6,13), (9,10), (9,14), (10,11), # | net.line
      (12,13), (13,14),                                # /
      (4,7), (4,9), (5,6), (7,8), (7,9)                # - net.trafo
      ]

def pickle_load(fp):
    """
    Returns the contents of a pickled file.
    """
    with open(fp, "rb") as f:
        return load(f)

def plt_config(x_vec, x=txt_discrete, y=txtf_measurement(0,0)):
    plt.xlim([0, x_vec.shape[1]-1])
    plt.legend(facecolor="white")
    plt.xlabel(x)
    plt.ylabel(y, multialignment="center")
    plt.tight_layout()

def plot_measurements(zh, zp, sensors, fn, ext=".pdf"):
    for sensor in sensors:
        plt.figure()
        plt.plot(zh[sensor, :], "b", marker="*", label="healthy", linewidth=2)
        plt.plot(zp[sensor, :], 'r--', marker="s", label="perturbed", linewidth=2)
        plt_config(zh, y=txtf_measurement(*ps[sensor]))
        plt.savefig(unique_filename(fn + f"_z{sensor}_hp" + ext))
        plt.close()

def plot_states(x_est, x2_est, nodes, fn, ext=".pdf"):
    for node in nodes:
        plt.figure()
        plt.plot(x_est[node, :], "b", marker="*", label="healthy", linewidth=2)
        plt.plot(x2_est[node, :], 'r--', marker="s", label="perturbed", linewidth=2)
        plt_config(x_est, y=txtf_state(node+1))
        plt.savefig(unique_filename(fn + f"_x{node}_hp" + ext))
        plt.close()

def plot_residuals(r, r2, fn, ext=".pdf"):
    plt.figure()
    plt.plot(np.nanmax(np.abs(r), axis=0), "b", marker="*", linewidth=2, label="healthy")
    plt.plot(np.nanmax(np.abs(r2), axis=0), "r--", marker="s", linewidth=2, label="perturbed")
    plt_config(r, y=txt_residuals)
    plt.savefig(unique_filename(fn + ext))
    plt.close()

def norm_1():
    fn = folder + "norm_1"
    ext = ".pdf"

    td = pickle_load("data/dc_norm1")
    z, z2, z3 = td["zs"]
    r, r2, r3 = td["rs"]

    plt.figure()
    plt.plot(z[1, :], "b", marker="*", label="healthy", linewidth=2)
    plt.plot(z3[1, :], 'r--', marker="s", label="naive", linewidth=2)
    plt_config(z, y=txtf_measurement(*ps[1]))
    plt.savefig(unique_filename(fn + "_z1_hn" + ext))
    plt.close()

    plt.figure()
    plt.plot(np.nanmax(np.abs(r), axis=0), "b", marker="*", linewidth=2, label="healthy")
    plt.plot(np.nanmax(np.abs(r3), axis=0), "r--", marker="s", linewidth=2, label="naive")
    plt.plot([3]*30, "g", linewidth=4, label="anomaly threshold")
    plt_config(r, y=txt_residuals)
    plt.savefig(unique_filename(fn + "_r_hn" + ext))
    plt.close()

    plot_measurements(z, z2, [1,2,3,4], fn)
    plot_residuals(r, r2, fn + "_r_hp")

def small_ubiquitous():
    fn = folder + "small_ubiquitous"

    td = pickle_load("data/dc_small_ubiquitous")
    z, z2 = td["zs"]
    r, r2 = td["rs"]

    plot_measurements(z, z2, range(z.shape[0]), fn)
    plot_residuals(r, r2, fn + "_r_hp")

def targeted_norm_1():
    fn = folder + "targeted_norm_1"

    td = pickle_load("data/dc_targeted_norm1")
    z, z2 = td["zs"]
    x_est, x2_est = td["x_ests"]
    r, r2 = td["rs"]
    a = td["aa"][0]
    c = x2_est[:,10] - x_est[:,10]

    sensors = [s for s in sensors if np.max(np.abs(a[s])) >= 1e-5]
    plot_measurements(z, z2, sensors, fn)
    plot_states(x_est, x2_est, [1,2,3,4], fn)
    plot_residuals(r, r2, fn + "_r_hp")

    
def targeted_small_ubiquitous():
    fn = folder + "targeted_small_ubiquitous"

    td = pickle_load("data/dc_targeted_small_ubiquitous-2")
    z, z2 = td["zs"]
    x_est, x2_est = td["x_ests"]
    r, r2 = td["rs"]
    a = td["aa"][0]
    c = x2_est[:,10] - x_est[:,10]

    plot_measurements(z, z2, range(z.shape[0]), fn)
    plot_states(x_est, x2_est, range(x_est.shape[0]), fn)
    plot_residuals(r, r2, fn + "_r_hp")

def targeted_matching_pursuit():
    fn = folder + "targeted_matching_pursuit"

    td = pickle_load("data/dc_targeted_matching_pursuit")
    z, z2 = td["zs"]
    x_est, x2_est = td["x_ests"]
    r, r2 = td["rs"]
    a = td["aa"][0]
    c = x2_est[:,10] - x_est[:,10]

    plot_measurements(z, z2, range(z.shape[0]), fn)
    plot_states(x_est, x2_est, range(x_est.shape[0]), fn)
    plot_residuals(r, r2, fn + "_r_hp")

def random_matrix_sc1a():
    fn = "figures/random_matrix_sc1a"
    ext = ".pdf"
    td = pickle_load("data/dc_random_matrix_sc1a")
    r, r2 = td["rs"]
    dfrom, dto = (1075, 1225)
    domain = list(range(dfrom, dto))

    plt.figure()
    plt.plot(domain, np.nanmax(np.abs(r), axis=0)[domain], "b", marker="*", linewidth=2, label="healthy")
    plt.plot(domain, np.nanmax(np.abs(r2), axis=0)[domain], "r--", marker="s", linewidth=2, label="perturbed")
    plt_config(r, y=txt_residuals)
    plt.xlim(dfrom, dto)
    plt.savefig(unique_filename(fn + "_r_hp" + ext))
    plt.close()

def random_matrix_sc1b():
    fn = "figures/random_matrix_sc1b"
    ext = ".pdf"
    td = pickle_load("data/dc_random_matrix_sc1b")
    r, r2 = td["rs"]

    dfrom, dto = (1075, 1225)
    domain = list(range(dfrom, dto))
    plt.figure()
    plt.plot(domain, np.nanmax(np.abs(r), axis=0)[domain], "b", marker="*", linewidth=2, label="healthy")
    plt.plot(domain, np.nanmax(np.abs(r2), axis=0)[domain], "r--", marker="s", linewidth=2, label="perturbed")
    plt_config(r, y=txt_residuals)
    plt.xlim(dfrom, dto)
    plt.savefig(unique_filename(fn + "_r_hp" + ext))
    plt.close()

def random_matrix_sc2():
    fn = "figures/random_matrix_sc2"
    ext = ".pdf"
    td = pickle_load("data/dc_random_matrix_sc2")
    z, z2 = td["zs"]
    x_est, x2_est = td["x_ests"]
    r, r2 = td["rs"]

    dfrom, dto = (1075, 1225)
    domain = list(range(dfrom, dto))

    a = td["aa"][0]
    c = (x2_est[:, 1100] - x_est[:, 1100])

    for sensor in [0, 6]: # most dominant
        plt.figure()
        plt.plot(domain, z[sensor, domain], "b", marker="*", label="healthy", linewidth=2)
        plt.plot(domain, z2[sensor, domain], 'r--', marker="s", label="perturbed", linewidth=2)
        plt_config(z, y=txtf_measurement(*ps[sensor]))
        plt.xlim(dfrom, dto)
        plt.savefig(unique_filename(fn + f"_z{sensor}_hp" + ext))
        plt.close()

    for node in [1, 12]: # most dominant
        plt.figure()
        plt.plot(domain, x_est[node, domain], "b", marker="*", label="healthy", linewidth=2)
        plt.plot(domain, x2_est[node, domain], 'r--', marker="s", label="perturbed", linewidth=2)
        plt_config(z, y=txtf_state(node+1))
        plt.xlim(dfrom, dto)
        plt.savefig(unique_filename(fn + f"_x{node}_hp" + ext))
        plt.close()

    plt.figure()
    plt.plot(domain, np.nanmax(np.abs(r), axis=0)[domain], "b", marker="*", linewidth=2, label="healthy")
    plt.plot(domain, np.nanmax(np.abs(r2), axis=0)[domain], "r--", marker="s", linewidth=2, label="perturbed")
    plt_config(r, y=txt_residuals)
    plt.xlim(dfrom, dto)
    plt.savefig(unique_filename(fn + "_r_hp" + ext))
    plt.close()

def modal_decomposition():
    fn = "figures/modal_decomposition"
    ext = ".pdf"
    td = pickle_load("data/dc_modal_decomposition")
    z, z2 = td["zs"]
    x_est, x2_est = td["x_ests"]
    r, r2 = td["rs"]

    dfrom, dto = (1075, 1225)
    domain = list(range(dfrom, dto))

    a = td["aa"][0]
    c = (x2_est[:, 1100] - x_est[:, 1100])

    for sensor in [0, 6]: # most dominant
        plt.figure()
        plt.plot(domain, z[sensor, domain], "b", marker="*", label="healthy", linewidth=2)
        plt.plot(domain, z2[sensor, domain], 'r--', marker="s", label="perturbed", linewidth=2)
        plt_config(z, y=txtf_measurement(*ps[sensor]))
        plt.xlim(dfrom, dto)
        plt.savefig(unique_filename(fn + f"_z{sensor}_hp" + ext))
        plt.close()

    for node in [3, 11]: # most dominant
        plt.figure()
        plt.plot(domain, x_est[node, domain], "b", marker="*", label="healthy", linewidth=2)
        plt.plot(domain, x2_est[node, domain], 'r--', marker="s", label="perturbed", linewidth=2)
        plt_config(z, y=txtf_state(node+1))
        plt.xlim(dfrom, dto)
        plt.savefig(unique_filename(fn + f"_x{node}_hp" + ext))
        plt.close()

    plt.figure()
    plt.plot(domain, np.nanmax(np.abs(r), axis=0)[domain], "b", marker="*", linewidth=2, label="healthy")
    plt.plot(domain, np.nanmax(np.abs(r2), axis=0)[domain], "r--", marker="s", linewidth=2, label="perturbed")
    plt_config(r, y=txt_residuals)
    plt.xlim(dfrom, dto)
    plt.savefig(unique_filename(fn + "_r_hp" + ext))
    plt.close()

def random_noise_norm1():
    fn = "figures/random_noise_norm1"
    td = pickle_load("data/dc_random_noise_norm1_p2")
    r, r2, r3 = td["rs"]

    plot_residuals(r, r2, fn + "_r_hp")
    plot_residuals(r, r3, fn + "_r_hp")

def random_noise_random_matrix():
    fn = folder + "random_noise_random_matrix"
    ext = ".pdf"

    td = pickle_load("data/dc_random_noise_random_matrix-2")
    Ts = td["Ts"]
    rs = td["rs"]

    dfrom, dto = (900, 2100)
    domain = list(range(dfrom, dto))

    for i in range(len(Ts)):
        plt.figure()
        plt.plot(domain, np.nanmax(np.abs(rs[0]), axis=0)[domain], "b", marker="*", linewidth=2, label="healthy")
        plt.plot(domain, np.nanmax(np.abs(rs[i+1]), axis=0)[domain], "r--", marker="s", linewidth=2, label="perturbed")
        plt_config(rs[0], y=txt_residuals)
        plt.xlim(dfrom, dto)
        plt.savefig(unique_filename(fn + f"_T{Ts[i]}" + ext))
        plt.close()

def random_noise_modal_decomposition():
    fn = folder + "random_noise_modal_decomposition"

    td = pickle_load("data/dc_random_noise_modal_decomposition-2")
    Ts = td["Ts"]
    rs = td["rs"]

    print(td["aaa"])

    for i in range(len(Ts)):
        plot_residuals(rs[0], rs[i+1], fn+f"_T{Ts[i]}")

def computational_complexity():
    fn = folder + "computational_complexity"
    ext = ".pdf"
    tds = [
        pickle_load("data/dc_computational_complexity_norm1"),
        pickle_load("data/dc_computational_complexity_small_ubiquitous"),
        pickle_load("data/dc_computational_complexity_targeted_norm1"),
        pickle_load("data/dc_computational_complexity_targeted_small_ubiquitous"),
        pickle_load("data/dc_computational_complexity_targeted_matching_pursuit"),
        pickle_load("data/dc_computational_complexity_random_matrix"),
    ]
    labels = [
        "LE",
        "SU",
        "TLE",
        "TSU",
        "OMP",
        "RMT"
    ]

    td_md = [pickle_load("data/dc_computational_complexity_modal_decomposition")]
    label_md = ["ICA"]

    times = []
    for td in tds:
        times.append([out[2] for out in td["outcomes"]])

    plt.figure()
    ax = plt.gca()
    ax.violinplot([times[i] for i in range(len(tds))])
    ax.set_xticks(range(1, len(tds)+1), labels[:len(tds)])
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.savefig(unique_filename(fn + "_p1" + ext))
    plt.close()

    times = []
    for td in td_md:
        times.append([out[2] for out in td["outcomes"]])
        
    plt.figure()
    ax = plt.gca()
    ax.violinplot(times[0])
    ax.set_xticks([1], label_md)
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.savefig(unique_filename(fn + "_p2" + ext))
    plt.close()

def staleness():
    fn = folder + "staleness"
    ext = ".pdf"
    td_rm = pickle_load("data/dc_staleness_random_matrix")
    td_md = pickle_load("data/dc_staleness_modal_decomposition")
    td_ac = pickle_load("data/ac_staleness_norm1-2")

    dfrom, dto = (475, 625)
    domain = list(range(dfrom, dto))

    r,r2 = td_rm["rs"]
    plt.figure()
    plt.plot(domain, np.nanmax(np.abs(r), axis=0)[domain], "b", marker="*", linewidth=2, label="healthy")
    plt.plot(domain, np.nanmax(np.abs(r2), axis=0)[domain], "r--", marker="s", linewidth=2, label="perturbed")
    plt_config(r, y=txt_residuals)
    plt.xlim(dfrom, dto)
    plt.savefig(unique_filename(fn + "_random_matrix" + ext))
    plt.close()

    r,r2 = td_md["rs"]
    plt.figure()
    plt.plot(domain, np.nanmax(np.abs(r), axis=0)[domain], "b", marker="*", linewidth=2, label="healthy")
    plt.plot(domain, np.nanmax(np.abs(r2), axis=0)[domain], "r--", marker="s", linewidth=2, label="perturbed")
    plt_config(r, y=txt_residuals)
    plt.xlim(dfrom, dto)
    plt.savefig(unique_filename(fn + "_modal_decomposition" + ext))
    plt.close()

    r,r2 = td_ac["rs"]
    plot_residuals(r, r2, fn + "_ac_norm1")

def computational_complexity_scalability():
    fn = folder + "computational_complexity_scalability_"
    ext = ".pdf"
    tds = [
        pickle_load("data/dc_computational_complexity_14_summary"),
        pickle_load("data/dc_computational_complexity_39_summary"),
        pickle_load("data/dc_computational_complexity_118_summary"),
        pickle_load("data/dc_computational_complexity_200_summary"),
        pickle_load("data/dc_computational_complexity_1354_summary"),
    ]
    times = {} # {test_model: {size: [times]}}
    for test in tds[0]:
        times[test] = []
        for i, td in enumerate(tds):
            if test in td:
                times[test].append([t[2] for t in td[test]["outcomes"]]) # t = (k, alpha, time)
    
    means = []
    for test in tds[0]:
        for i in range(len(times[test])):
            means.append(round(np.mean(times[test][i]),5))
        print(f"{test}: {means[-len(times[test]):]}")

    labels = ["IEEE 14", "IEEE 39", "IEEE 118", "Illinois 200", "PEGASE 1354"]
    filenames = list((map(lambda s: fn + s + ext, tds[0].keys())))

    for i, test in enumerate(tds[0]):
        plt.figure()
        ax = plt.gca()
        ax.violinplot(times[test])
        ax.set_xticks(range(1, len(times[test])+1), labels[:len(times[test])])
        plt.ylabel("Time (s)")
        plt.tight_layout()
        #plt.show()
        plt.savefig(unique_filename(filenames[i]))
        plt.close()

def computational_complexity_double_measurements():
    fn = folder + "computational_complexity_double_measurements_"
    ext = ".pdf"
    tds = [
        pickle_load("data/dc_computational_complexity_double_14_summary"),
        pickle_load("data/dc_computational_complexity_double_39_summary"),
        pickle_load("data/dc_computational_complexity_double_118_summary"),
    ]
    times = {} # {test_model: {size: [times]}}
    for test in tds[0]:
        times[test] = []
        for i, td in enumerate(tds):
            if test in td:
                times[test].append([t[2] for t in td[test]["outcomes"]]) # t = (k, alpha, time)
    
    means = []
    for test in tds[0]:
        for i in range(len(times[test])):
            means.append(round(np.mean(times[test][i]),5))
        print(f"{test}: {means[-len(times[test]):]}")

    labels = ["IEEE 14", "IEEE 39", "IEEE 118", "Illinois 200", "PEGASE 1354"]
    filenames = list((map(lambda s: fn + s + ext, tds[0].keys())))

    for i, test in enumerate(tds[0]):
        plt.figure()
        ax = plt.gca()
        ax.violinplot(times[test])
        ax.set_xticks(range(1, len(times[test])+1), labels[:len(times[test])])
        plt.ylabel("Time (s)")
        plt.tight_layout()
        plt.savefig(unique_filename(filenames[i]))
        plt.close()


if __name__ == "__main__":
    import seaborn as sns
    sns.set_context("paper", font_scale=3.5)
    sns.set_style("darkgrid")
    sns.set_palette("deep")
    sns.set(font='sans-serif')
    plt.rcParams['figure.dpi'] = 140
    #plt.rcParams['figure.facecolor'] = 'grey'
    plt.rcParams['figure.edgecolor'] = 'grey'
    #plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.linestyle'] = 'dashed'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.color'] = 'black'

    targeted_small_ubiquitous()
