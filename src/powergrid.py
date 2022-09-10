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
    # constants. They are set through the given input parameters.
    network            = None # 'network_id'
    H                  = None # 'network_id'
    measurement_factor = 1/10 # 'measurement_factor', optional
    noise_factor       = 1/50 # 'noise_factor', optional
    anomaly_threshold  = 3    # 'anomaly_threshold', optional
    
    # These variables are updated (and overwritten) upon calling the given
    # functions.
    data_generation_strategy = None # create_measurements()
    z_buffer                 = None # create_measurements()
    cov                      = None # create_measurements(); used when generating noise and normalizing residuals
    x_est                    = None # estimate_state()
    residuals_normalized     = None # calculate_residue()

    def __init__(self, network_id, measurement_factor=None, noise_factor=None, 
                 anomaly_threshold=None):
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
        line = self.network.line
        rows = line.shape[0]
        connections = rows + self.network.trafo.shape[0]
        self.H = np.zeros((connections, self.network.bus.shape[0]))

        # Some connections are found in {from, to}_bus, others are found in
        # trafo.{hv, lv}_bus.
        for from_index, to_index, from_bus, to_bus, j in (
            (0, rows, 
                line.from_bus.values, line.to_bus.values, 
                lambda i: i), 
            (rows+1, connections, 
                self.network.trafo.hv_bus.values, self.network.trafo.lv_bus.values,
                lambda i: i - rows)
            ):
            
            for i in range(from_index, to_index):
                from_j = from_bus[j(i)]
                to_j = to_bus[j(i)]
                power_flow = 1 / A_real[from_j, to_j]
                self.H[i, from_j] = power_flow
                self.H[i, to_j]  = -power_flow

        if noise_factor:
            self.noise_factor = noise_factor
        if measurement_factor:
            self.measurement_factor = measurement_factor
        if anomaly_threshold:
            self.anomaly_threshold = anomaly_threshold
        

    def create_measurements(self, T, data_generation_strategy):
        """
        Generates T measurements according to
            z = H*x + n
        where H is derived from a preset network configuration, x is generated
        based on the chosen data_generation_strategy, and n represents random
        noise.

        The argument data_generation_strategy can be either 1 or 2.

        Returns a [X x T] numpy array -- where X is the number of measurements
        for each time step -- which is also stored in self.z_buffer.
        """
        x_temp = self.network.res_bus.va_degree
        if data_generation_strategy == 1:
            # state vector x_base
            x_base = x_temp.to_numpy()
            # Dx ~ N(0,Cov_Mat)
            mean = np.zeros(x_base.shape)
            self.cov = np.eye(x_base.shape[0]) * self.measurement_factor
            delta_x_mat = np.transpose(np.random.multivariate_normal(mean, self.cov, size=T))
            x_base_mat = np.repeat(x_base.reshape((x_base.shape[0], -1)), T, axis=1)
            x_t_mat = x_base_mat + delta_x_mat

        elif data_generation_strategy == 2:
            x_t = x_temp.to_numpy()
            x_t_mat = x_t.reshape((x_t.shape[0], -1))
            p_mw = self.network.load.p_mw
            mean = np.zeros(p_mw.shape) * self.measurement_factor
            self.cov = np.eye(p_mw.shape[0])    
            for t in range(1, T):
                delta_load = np.random.multivariate_normal(mean, cov)
                p_mw.add(pd.Series(delta_load)) 
                pp.rundcpp(self.network)
                x_t = x_temp.to_numpy().reshape((x_t.shape[0], -1))
                x_t_mat = np.hstack((x_t_mat, x_t))
        else:
            raise ValueError(f"Invalid data_generation_strategy: {data_generation_strategy}")

        self.data_generation_strategy = data_generation_strategy

        #z_t = H @ x_t + noise
        mean = np.zeros(self.H.shape[0])
        cov = np.eye(self.H.shape[0]) * self.noise_factor
        noise_mat = np.transpose(np.random.multivariate_normal(mean, cov, size=T))
        z_t_mat = self.H @ x_t_mat + noise_mat # @ is infix for np.matmul()
        self.z_buffer = z_t_mat
        return z_t_mat

    def estimate_state(self, z=None):
        """
        Calculates state estimations based on the given network configuration
        and observed measurements according to:
            x_est = (H_est_transpose * H_est)^-1 * H_est_transpose * z
        where H_est is equal to H with the exception that the first column
        is removed (under the assumption that the remaining state variables
        x[1:, :] are relative to x[0, :]).
        """
        z = self._load_var("z_buffer", z, err_msg="Cannot estimate state without measurements.")

        H_est = np.copy(self.H[:, 1:])
        H_est_transpose = np.transpose(H_est)
        self.x_est = np.linalg.inv(H_est_transpose @ H_est) @ H_est_transpose @ z

        # Prepend zeroes as the first measurement, since the latter ones are
        # measured relative to it.
        self.x_est = np.vstack((np.zeros((self.x_est.shape[1])), self.x_est))

        return self.x_est

    def calculate_normalized_residuals(self, z=None, x_est=None):
        """
            Calculates residual vectors, which represent the 
            difference between observed measurements and estimated measurements, 
            according to
                r = (z - H * x_est).
            for each time step (= column) in z.
            These residuals are then normalized and returned.
        """
        err_msg = "Cannot calculate residue without "
        z = self._load_var("z_buffer", z, err_msg + "z")
        x_hat = self._load_var("x_est", x_est, err_msg + "x_est")

        r = z - self.H @ x_hat
        self.residuals_normalized = r / np.sqrt(self.cov[0, 0]) # check if abs(measurement)>3
        return self.residuals_normalized

    def generate_anomalous_measurements(self):
        """
            todo
        """
        pass

    def check_for_anomalies(self, residuals_normalized=None):
        """
            Checks whether a matrix of residual vectors contain anomalies.
            An anomalous measurement exists when a normalized vector has a
            value exceeding 3.
            A list of indexes of anomalous values is returned in the format
                [(i1, j1), (i2, j2), ...],
            where 'i' represents a particular measurement and 'j' a point in time.
        """
        r = self._load_var("residuals_normalized", residuals_normalized)

        anomalous_indexes = []

        # Only iterate through the matrix if an anomaly is present
        if r.max() > self.anomaly_threshold:
            with np.nditer(r, flags=["multi_index"]) as it:
                for value in it:
                    if value > self.anomaly_threshold:
                        i, j = it.multi_index
                        anomalous_indexes.append((i,j))
                    
        return anomalous_indexes

    def _load_var(self, default_key, arg, err_msg="Empty default_key: "):
        """
        Helper function that:
            - if 'arg' is None, returns the value stored in the attribute given
              by 'default_key': getattr(self, default_key)
            - otherwise, 'arg' is returned the attribute is set to this value.
        If 'arg' is None and the attribute does not have a value (or does not exist),
        then a ValueError exception is thrown.
        """
        if arg is None:
            if not default_key in dir(self) or getattr(self, default_key) is None:
                raise ValueError(err_msg + default_key)
            return getattr(self, default_key)
        setattr(self, default_key, arg)
        return arg

if __name__ == "__main__":
    print("testing: case 14")
    nw = PowerGrid(14) # Alternative arguments: "IEEE-14", "14"
    z = nw.create_measurements(1000, 1) # 'x' measurements using data strategy 'y'
    print(f"Created measurements {z.shape} with strategy 1:", z, sep="\n")
    x_est = nw.estimate_state() # Arguments can be provided to override stored values
    print("Estimated state:", x_est, x_est.shape, sep="\n")
    r = nw.calculate_normalized_residuals()
    print("Normalized residuals:", r, r.shape, sep="\n")
    anomalies = nw.check_for_anomalies()
    print("Anomalies:", [(i, r[i[0], i[1]]) for i in anomalies])