import pandapower as pp
import pandapower.networks as ppn
import numpy as np
import pandas as pd

class PowerGrid():
    """
    Logical representation of a (preset) network, with methods for state estimation
    and emulation of anomalous data. 
    """
    # These variables are created upon initialization, and could be considered
    # constants.
    network = None  # Created upon initialization.
    H = None        # Created upon initialization.
    
    # These variables are updated (and overwritten) upon calling the given
    # functions.
    data_generation_strategy = None # create_measurements()
    z_buffer                 = None # create_measurements()
    x_estimate               = None # estimate_state()

    def __init__(self, network_id):
        """
        Available networks include "IEEE-X", where X is one of [14, 30, 57, 118].
        The selected network will be loaded and stored in self.network.
        """
        if type(network_id) == type(int()):
            network_id = str(network_id)
        elif "IEEE-" in network_id: # Since all (currently) supported networks have the same prefix
            network_id = network_id[network_id.index("IEEE-")+5:]

        cases = {
            "14": ppn.case14,
            "30": ppn.case30,
            "57": ppn.case57,
            "118": ppn.case118,
        }
        if not network_id in cases:
            raise NotImplementedError(f"Unsupported network configuration: {network_id}")

        # Fetch network configuration
        self.network = cases[network_id]()
        pp.rundcpp(self.network)

        # P_from_node_i_to_node_j
        # Pij = (1/bij)*(x[i]-x[j])
        # H[line_id,i] = 1/bij
        # H[line_id,j] = -1/bij
        A_real = np.real(self.network._ppc['internal']['Bbus'].A)
        self.H = np.zeros((self.network.line.shape[0], self.network.bus.shape[0]))
        from_bus_values = self.network.line.from_bus.values
        to_bus_values = self.network.line.to_bus.values
        for i in range(0, self.network.line.shape[0]):
            power_flow = 1/A_real[from_bus_values[i], to_bus_values[i]]
            self.H[i, from_bus_values[i]] = power_flow
            self.H[i, to_bus_values[i]]  = -power_flow

    def create_measurements(self, T, data_generation_strategy):
        """
        Generates T measurements according to
            z = H*x + n
        where H is derived from a preset network configuration, x is generated
        based on the chosen data_generation_strategy, and n represents random
        noise.

        The argument data_generation_strategy can be either 1 or 2.

        Returns a [(X+1) x T] numpy array, and also stores it in self.z_buffer.
        """
        x_temp = self.network.res_bus.va_degree
        if data_generation_strategy == 1:
            # state vector x_base
            x_base = x_temp.to_numpy()
            # Dx ~ N(0,Cov_Mat)
            mean = np.zeros(x_base.shape)
            cov = np.eye(x_base.shape[0])
            delta_x_mat = np.transpose(np.random.multivariate_normal(mean, cov, size=T))
            x_base_mat = np.repeat(x_base.reshape((x_base.shape[0],-1)), T, axis=1)
            x_t_mat = x_base_mat + delta_x_mat

        elif data_generation_strategy == 2:
            x_t = x_temp.to_numpy()
            x_t_mat = x_t.reshape((x_t.shape[0],-1))
            for t in range(1, T):
                p_mw = self.network.load.p_mw
                mean = np.zeros(p_mw.shape)
                cov = np.eye(p_mw.shape[0])    
                delta_load = np.random.multivariate_normal(mean, cov)
                p_mw.add(pd.Series(delta_load)) 
                pp.rundcpp(self.network)
                x_t = x_temp.to_numpy().reshape((x_t.shape[0],-1))
                x_t_mat = np.hstack((x_t_mat, x_t))
        else:
            raise ValueError(f"Invalid data_generation_strategy: {data_generation_strategy}")

        self.data_generation_strategy = data_generation_strategy

        #z_t = H @ x_t + noise
        mean = np.zeros(self.H.shape[0])
        cov = np.eye(self.H.shape[0])/10
        noise_mat = np.transpose(np.random.multivariate_normal(mean, cov, size=T))
        z_t_mat = np.matmul(self.H, x_t_mat) + noise_mat
        print(f"Generated {z_t_mat.shape}")
        return z_t_mat

    def estimate_state(self, z=None):
        """
        Calculates state estimations based on the given network configuration
        and observed measurements according to
            x_hat = inv(H_transpose * H) * H_transpose * z.
        """
        if not z:
            if not self.z_buffer:
                raise ValueError("Cannot estimate state without measurements.")
            z = self.z_buffer
            

    def calculate_residue(self, z=None, xhat=None):
        """
            Calculates a residual vector, which represents the difference between
            observed measurements and estimated measurements, according to
            r = z - H * x_hat
        """
        pass


    def generate_anomalous_measurements(self):
        """
            todo
        """
        pass


if __name__ == "__main__":
    print("testing: case 14")
    nw = PowerGrid(14) # alternative args: "IEEE-14", "14"
    out = nw.create_measurements(3, 1)
    print(out)
    out2 = nw.create_measurements(3, 2)
    print(out)