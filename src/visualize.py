import tkinter as tk # gui
from tkinter import ttk # newer widgets

from pickle import load
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import use as plt_set_backend
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

global current_time, update_flag, label_string

plt_set_backend("TkAgg") # display plots in tkinter windows
root = tk.Tk()
root.title("DAT300 -- Power Grid Demo")
root.geometry("1500x900")
for i in range(4):
    root.columnconfigure(i, weight=1)
    root.rowconfigure(i, weight=1)
root.resizable(False, False)

current_time = -1
update_flag = True
label_string = tk.StringVar(root, "")

import seaborn as sns
sns.set_context("paper", font_scale=3.5)
sns.set_style("darkgrid")
sns.set_palette("deep")
sns.set(font='sans-serif')
plt.rcParams['figure.dpi'] = 100
#plt.rcParams['figure.facecolor'] = 'grey'
plt.rcParams['figure.edgecolor'] = 'grey'
#plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = 'dashed'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.color'] = 'black'

class Graph():
    description = ""
    data = []
    param = {} # figure_specific parameters
    _range = (0, 0) # (start, end)
    figure = None
    axes = None
    lines = {}
    bg = None

    # tkinter
    canvas = None

    def __init__(self, description, data, param={}):
        self.description = description
        self.data = data
        self.param = param
        self._range = (0, data[0].shape[0])
        self.create_figure()
        

    def create_figure(self):
        """
        Creates a basic figure using the first list in self.data.
        """
        self.figure = plt.figure(figsize=(5, 2))
        self.figure.set_layout_engine("constrained") # matplotlib 3.6+
        self.axes = self.figure.add_subplot()
        print(self._range)

        self.lines = [self.axes.plot(line, animated=True)[0] for line in self.data]
        if len(self.lines) > 1:
            self.lines[1].set_color("red")
            self.lines[1].set_linestyle("--")
        #plt.yticks([])
        self.axes.set_xlim(*self._range)
        #graph.figure.set_xdata(list(range(graph._range[1])))
        #plt.ylim(*plot_vars["y_lims"])
        self.canvas = FigureCanvasTkAgg(self.figure, root)

def search(graphs, substr):
    """
    Look through a list of graphs and return those whose description
    contains the given substring.
    """
    return [g for g in graphs if substr in g.description]

def norm_1_demo_graphs(filepath):
    """
    Creates and returns a list of graphs using data from demo_norm_1() from
    experiments_dc.py, as well as a function for updating them wrt. time.
    Takes as input a file path for the pickled data.
    """
    global current_time
    with open(filepath, "rb") as f:
        norm_1_data = load(f)
        print("norm_1", norm_1_data)
    graphs = []
    graphs.append(Graph("z1", [
        norm_1_data["z"][0, :],
        norm_1_data["z_a"][0, :]
        ]))
    graphs.append(Graph("z2", [
        norm_1_data["z"][1, :],
        norm_1_data["z_a"][1, :]
        ]))
    graphs.append(Graph("z3", [
        norm_1_data["z"][2, :],
        norm_1_data["z_a"][2, :]
        ]))
    graphs.append(Graph("z4", [
        norm_1_data["z"][3, :],
        norm_1_data["z_a"][3, :]
        ]))
    graphs.append(Graph("x1", [
        norm_1_data["x_est"][0, :],
        norm_1_data["x_est_a"][0, :]
        ]))
    graphs.append(Graph("x2", [
        norm_1_data["x_est"][1, :],
        norm_1_data["x_est_a"][1, :],
        ]))
    graphs.append(Graph("x3", [
        norm_1_data["x_est"][2, :],
        norm_1_data["x_est_a"][2, :]
        ]))
    graphs.append(Graph("x4", [
        norm_1_data["x_est"][3, :],
        norm_1_data["x_est_a"][3, :],
        ]))
    graphs.append(Graph("r",  [
        np.nanmax(np.abs(norm_1_data["r"]), axis=0),
        np.nanmax(np.abs(norm_1_data["r_a"]), axis=0),
        [3] * 30
        ]))

    z0 = search(graphs, "z")[0]
    z3 = search(graphs, "z")[-1]
    x0 = search(graphs, "x")[0]
    x3 = search(graphs, "x")[-1]
    r = search(graphs, "r")[0]

    r.figure.set_figheight(4)

    for g in graphs:
        g.axes.xaxis.set_ticks([0,1,5,9,15,19,25,29])

    z0.axes.set_title("Measurements [p.u.]")
    x0.axes.set_title("Estimated states [radians]")
    r.axes.set_title("Residuals [p.u.]")

    for g in [z3, x3, r]:
        g.axes.set_xlabel("Discrete time steps")

    r.lines[0].set_label("Normal")
    r.lines[1].set_label("With injection")
    r.lines[2].set_label("Anomaly threshold")
    r.figure.legend(facecolor="white", loc="upper right", 
                    bbox_to_anchor=(1, 0.95, 0, 0))

    checkpoints = [4,10,14,20,24,29]

    def update_graphs(time_delta, auto=False):
        global current_time, update_flag, label_string
        print("update", current_time, time_delta, update_flag)

        # Prevent the function from being called too quickly
        if not update_flag and not auto:
            return
        
        update_flag = False

        if time_delta == 0: # Advance until the next checkpoint
            auto = True
            time_delta += 1

        if current_time + time_delta not in range(30):
            update_flag = True
            return

        # Since removing points take much longer than adding them,
        # set current_time to a time in checkpoints (or 0)
        if time_delta == -1:
            current_time = [0] + [t for t in checkpoints if t < current_time]
            current_time = current_time[-1]
        else:
            current_time += time_delta
            
        for g in graphs:
            if time_delta < 0:
                g.canvas.draw()
                # This takes a long time, but canvas.restore_from_region does
                # not appear to behave nicely with tkinter (the way it is
                # currently configured, at least), so the static background
                # will have to be redrawn when when removing data points.
                for i in range(len(g.data)):
                    g.lines[i].set_data(
                        list(range(current_time + 1)),
                        g.data[i][:current_time + 1]
                        )
                    g.axes.draw_artist(g.lines[i])
                    g.canvas.blit()
                

            else:
                for i in range(len(g.lines)):
                    g.lines[i].set_data(
                        list(range(max(0, current_time - 1), current_time + 1)),
                        g.data[i][max(0, current_time - 1): current_time + 1]
                        )
                    g.axes.draw_artist(g.lines[i])
                    g.canvas.blit()

        for g in graphs:
            g.canvas.flush_events()

        if auto and current_time not in checkpoints:
            root.after(200, lambda: update_graphs(1, True))
        else:
            update_flag = True

        text = ""
        alpha = lambda i: round(0.5 + 0.1 * min(4,current_time - i), 1)
        if current_time in range(5):
            text = ("Time steps 0-4:\n" +
                    "No injection\n \n \n \n ")
        elif current_time in range(5, 11):
            text = ("Time steps 5-9:\n" +
                    "Naive injection\n" +
                    "Target: second sensor\n" +
                    "Affected: second\n" +
                   f"α = {alpha(5)}\n ")
        elif current_time in range(11, 15):
            text = ("Time steps 10-14:\n"+
                    "no injection\n \n \n \n ")
        elif current_time in range(15, 21):
            text = ("Time steps 15-19:\n" +
                    "Least effort injection\n" +
                    "Target: second sensor\n" +
                    "Affected: second and fourth\n" +
                   f"α = {alpha(15)}\n ")
        elif current_time in range(21, 25):
            text = ("Time steps 20-24:\n" +
                    "No injection\n \n \n \n ")
        elif current_time in range(25, 30):
            text = ("Time steps 25-29:\n" +
                    "Least effort injection\n" +
                    "Target: second sensor\n" +
                    "Affected: first and second\n" +
                   f"α = {alpha(15)}\n" +
                    "The fourth sensor is read-only")
        label_string.set(text)


    return graphs, update_graphs

def init_grid(graphs):
    r = 0
    for g in graphs[:-1]:
        g.canvas.get_tk_widget().grid(row=r % 4, column=r // 4, sticky=tk.NS)
        r += 1

    graphs[-1].canvas.get_tk_widget().grid(row=r % 4, column=r // 4, rowspan=2, sticky=tk.NS)
    r += 2

    return r

# TODO:
# - buttons to switch between sets of graphs
# - additional semantics:
#   . targeted small ubiquitous (c1 += [0.01, 0.05])
#   . random matrix

def init():
    graphs, update = norm_1_demo_graphs("data/dc_norm1_demo")
    
    r = init_grid(graphs)
    label = ttk.Label(root, textvariable=label_string, 
                      font="Calibri 20", anchor="w", justify="left")
    label.grid(row = r % 4, column = r // 4, sticky = tk.SW, padx = 50)

    update(1)
    return update

if __name__ == "__main__":
    update = init()
    root.protocol("WM_DELETE_WINDOW", exit)
    root.bind("<Left>",  lambda ev: update(-1))
    root.bind("<Right>", lambda ev: update(1))
    root.bind("<Down>", lambda ev: update(min(-1, -current_time)))
    root.bind("<space>", lambda ev: update(0))
    root.mainloop()
