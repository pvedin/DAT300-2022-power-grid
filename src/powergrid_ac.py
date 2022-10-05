print("Loading libraries...")
import pandapower as pp
import pandapower.networks as ppn
from pandapower.pypower.dSbr_dV import dSbr_dV
from pandas import Series
from scipy import sparse
import numpy as np
import pyomo.environ as pyo
import idaes
from time import time

print("Loaded!")

class AnomalyModels():

    def least_effort_norm_1(**cvs):
        net = cvs["net"].network
        H = cvs["H"]
        x_est = cvs["x_est"]
        z_x_est = cvs["z_x_est"].reshape(-1)

        k = 1
        delta_a = 0.1 # in reality, a change by 0.1 is 0.1*100 MW = 10 MW
        a_lower_bound, a_upper_bound, c_lower_bound, c_upper_bound = -1000, 1000, -1000, 1000

        m = pyo.ConcreteModel()

        m.a_num = range(H.shape[0])
        m.c_num = range(H.shape[1]+1)
        m.a = pyo.Var(m.a_num, domain=pyo.Reals, initialize=0)
        m.c = pyo.Var(m.c_num, domain=pyo.Reals, initialize=0)
        m.a_pos = pyo.Var(m.a_num, domain=pyo.NonNegativeReals, initialize=0)
        m.a_neg = pyo.Var(m.a_num, domain=pyo.NonPositiveReals, initialize=0)

        m.abs_a = pyo.ConstraintList()
        for i in m.a_num:
            m.abs_a.add(m.a[i] == m.a_pos[i]+m.a_neg[i]) 
            
        def compute_h_m(x_est, m, net):  
            # x_est is the final estimate for a given point in time, x_hat[:,t]
            x_est_a = np.asarray([np.angle(x_est[i]) + m.c[i] for i in range(0,net._ppc['bus'].shape[0])])
            x_est_m = np.asarray([np.abs(x_est[i-net._ppc['bus'].shape[0]]) + m.c[i] for i in range(net._ppc['bus'].shape[0],2*net._ppc['bus'].shape[0])])

            Z = net._ppc['branch'][:,2] + 1j*net._ppc['branch'][:,3]
            C = 1/Z

            tap = np.real(net._ppc['branch'][:,8])

            g = np.real(C)
            b = np.imag(C)

            bs_line = net._ppc['branch'][:,4].astype(float)/2

            Pij = []
            Qij = []

            for i in range(0, net._ppc['branch'].shape[0]):
                bi0 = np.real(net._ppc['branch'][i, 0]).astype(int)
                bi1 = np.real(net._ppc['branch'][i, 1]).astype(int)
                Pij.append(x_est_m[bi0]**2 * g[i]/tap[i]**2 
                        - x_est_m[bi0]*x_est_m[bi1]
                        *(g[i]*pyo.cos(x_est_a[bi0]-x_est_a[bi1])
                            + b[i]*pyo.sin(x_est_a[bi0]-x_est_a[bi1]))/tap[i])

                Qij.append(-x_est_m[bi0]**2 * ((b[i] + bs_line[i])/tap[i]**2)
                        - x_est_m[bi0]*x_est_m[bi1]
                        *(g[i]*pyo.sin(x_est_a[bi0]-x_est_a[bi1])
                            - b[i]*pyo.cos(x_est_a[bi0]-x_est_a[bi1]))/tap[i])

            Pij = np.asarray(Pij).reshape((net._ppc['branch'].shape[0],-1))
            Qij = np.asarray(Qij).reshape((net._ppc['branch'].shape[0],-1))    
            
            V_x_est = x_est_m[0]
            z_x_est = np.vstack((Pij,Qij)).reshape(-1)
            z_x_est = np.hstack((z_x_est,V_x_est))
            # z_x_est is h(x+c)
            return z_x_est 

        # a = h(x_hat+c) - h(x_hat) instead of a = Hc, where x_hat is the output of AC estimator

        m.stealthy_a = pyo.ConstraintList()
        for i in m.a_num:
            # z_x_est is the h(x_hat) from the output of the AC estimator
            # compute_h_m(x_est, m, net) is the h(x_hat + c), where c is a variable vector
            m.stealthy_a.add(m.a[i] == compute_h_m(x_est, m, net)[i] - np.real(z_x_est[i]))

        m.change_a = pyo.Constraint(expr = (m.a[k] == delta_a))

        m.change_c = pyo.Constraint(expr = (m.c[0] == 0))
            
        m.bounds_a = pyo.ConstraintList()
        for i in m.a_num:
            m.bounds_a.add(m.a[i] <= a_upper_bound)
            m.bounds_a.add(m.a[i] >= a_lower_bound)

        m.bounds_c = pyo.ConstraintList()
        for i in m.c_num:
            m.bounds_c.add(m.c[i] <= c_upper_bound)
            m.bounds_c.add(m.c[i] >= c_lower_bound)

        def abs_a_val(m):
            return sum(m.a_pos[i] - m.a_neg[i] for i in m.a_num)
            
        m.value = pyo.Objective(rule=abs_a_val, sense = pyo.minimize)    
            
        #m.pprint()

        optimizer = pyo.SolverFactory('ipopt')
        status_opti = optimizer.solve(m, tee=True)

        a_output = np.asarray([pyo.value(m.a[i]) for i in m.a_num])
        c_output = np.asarray([pyo.value(m.c[i]) for i in m.c_num])

        return a_output, c_output



class PowerGrid():
    """
    Logical representation of a (preset) network, with methods for state estimation
    and emulation of anomalous data. 
    """
    # These variables are created upon initialization, and could be considered
    # constants. They are set through the given input parameters.
    network            = None # 'network_id'
    H                  = None # 'network_id'
    measurement_factor = 1e-4 # 'measurement_factor', optional
    noise_factor       = 1e-3 # 'noise_factor', optional
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
        pp.runpp(self.network)

        if noise_factor:
            self.noise_factor = noise_factor
        if measurement_factor:
            self.measurement_factor = measurement_factor
        if anomaly_threshold:
            self.anomaly_threshold = anomaly_threshold

    def _h_x(self, x):
        Sf = []
        rs = self.network._ppc['branch'].shape[0]
        for i in range(0, rs): # to be refined
            Sf.append(x[self.network._ppc['branch'][i, 0].astype(int)]
                      * np.conj(self.network._ppc['internal']['Yf'][i,:]*x))
        Sf = np.asarray(Sf).reshape((rs,-1))
        z_P = np.real(Sf)
        z_Q = np.imag(Sf)
        if x.ndim == 1:
            z_V_mag = np.abs(x[0])
        elif x.ndim == 2:
            z_V_mag = np.abs(x[0,:])
        else:
            raise NotImplementedError()
        z_t_mat = np.vstack((z_P,z_Q))
        z_t_mat = np.vstack((z_t_mat,z_V_mag))
        return z_t_mat        

    def create_measurements(self, T, data_generation_strategy=1, env_noise=True):
        """
        Generates T measurements according to
            z = h(x) + n
        where h is derived from a preset network configuration, x is generated
        based on the chosen data_generation_strategy, and n represents random
        noise.

        The argument data_generation_strategy at the moment only be 1.

        env_noise will omit the noise added to each measurement if set to
        False; however, the noise added to the state vectors remains unchanged.

        Returns a [X x T] numpy array -- where X is the number of measurements
        for each time step -- which is also stored in self.z_buffer.
        """
        # state vector x_base (which is again (number of nodes, ) but a complex vector)
        if data_generation_strategy == 1:
            x_base = self.network.res_bus.vm_pu.to_numpy()*np.exp(1j*self.network.res_bus.va_degree.to_numpy()/180*np.pi)
            # Dx ~ N(0,Cov_Mat)
            mean = np.zeros(x_base.shape)
            self.cov = np.eye(x_base.shape[0])/1000 # to be refined in the future
            delta_x_mat = np.transpose(np.random.multivariate_normal(mean, self.cov**2, size=T))
            x_base_mat = np.repeat(x_base.reshape((x_base.shape[0],-1)), T, axis=1)
            x_t_mat = x_base_mat + delta_x_mat

        elif data_generation_strategy == 2:
            raise NotImplementedError()
        else:
            raise ValueError(f"Invalid data_generation_strategy: {data_generation_strategy}")

        self.data_generation_strategy = data_generation_strategy

        z_t_mat = self._h_x(x_t_mat)
        # in the DC case, we had 20 measurements for the 14-node bus since z_P (20 x 1)
        # in the AC case, we have 20 from z_P, 20 from z_Q, and 1 from z_V_mag -> 41

        if env_noise:
            rs = self.network._ppc['branch'].shape[0] * 2 + 1 # including measurements from z_Q and z_V_mag
            mean = np.zeros(rs)
            self.cov_noise = np.eye(rs) * self.noise_factor
            noise_mat = np.transpose(np.random.multivariate_normal(mean, self.cov_noise**2, size=T))
            z_t_mat += noise_mat
        
        return z_t_mat

    def estimate_state(self, z=None):
        """
        Calculates state estimations based on the given network configuration
        and observed measurements according to:
            x_est = TODO
        """
        # in DC state estimator: x_est = inv(H.T @ H) @ H.T @ z
        # in DC case: x = [theta_0, theta_1, ...]
        # in AC case: x = [theta_0, theta_1, ..., theta_13, mag_0, mag_1, ..., mag_13]
        # we need theta_0 = 0

        Hs = []
        x_ests = []
        z_x_ests = []

        for ts in range(z.shape[1]):
            # initial guess for state vector
            x_est = np.ones((len(self.network.bus)))

            # initially, all voltage magnitudes are assumed to be 1 and all voltage angles are assumed to 0
            residuals = 1

            count = 0
            while np.max(np.abs(residuals)) > 1e-2:
                count += 1
                print(count)

                # h(x_est) for the current x_est (which is just x_hat), equivalent to H*x_hat
                z_x_est = self._h_x(x_est)
                
                residuals = z[:, ts].reshape((-1,1)) - z_x_est


                dSf_dVa, dSf_dVm, _, _, _, _ = dSbr_dV(self.network._ppc['branch'], 
                                                    self.network._ppc['internal']['Yf'], 
                                                    self.network._ppc['internal']['Yt'], 
                                                    x_est)
                dV_dVa = 1j * np.diag(x_est)
                dV_dVa = sparse.csr_matrix(dV_dVa[0,:])
                dV_dVm = np.diag(x_est/np.abs(x_est))
                dV_dVm = sparse.csr_matrix(dV_dVm[0,:])
                
                H_P = sparse.hstack((np.real(dSf_dVa), np.real(dSf_dVm)))
                H_Q = sparse.hstack((np.imag(dSf_dVa), np.imag(dSf_dVm)))
                H_V = sparse.hstack((dV_dVa, dV_dVm))
                H = sparse.vstack((H_P, H_Q))
                H = sparse.vstack((H, H_V))
                H = sparse.csr_matrix(H)[:,1:]
                
                srs = self.network._ppc['bus'].shape[0]
                delta_x_est = sparse.linalg.spsolve(np.transpose(H) @ H, np.transpose(H) @ residuals)
                delta_x_est_a = delta_x_est[0:srs-1]
                delta_x_est_m = delta_x_est[srs-1:]

                x_est_a = np.angle(x_est[1:]) + delta_x_est_a
                x_est_a = np.hstack((0,x_est_a))
                x_est_m = np.abs(x_est) + delta_x_est_m
                x_est = x_est_m*np.exp(1j*x_est_a)

                print(np.max(np.abs(residuals)))

            Hs.append(H)
            x_ests.append(x_est)
            z_x_ests.append(z_x_est)
        
        return Hs, x_ests, z_x_ests

    def calculate_normalized_residuals(self, z=None, x_ests=None):
        """
            Calculates residual vectors, which represent the 
            difference between observed measurements and estimated measurements, 
            according to
                r = (z - H * x_est).
            for each time step (= column) in z.
            These residuals are then normalized and returned.
            TODO: fix as in powergrid_dc
        """
        z_x_ests = []
        for ts in range(z.shape[1]):
            z_x_ests.append(self._h_x(x_ests[ts]).reshape(-1))


        z_x_est = np.asarray(z_x_ests).transpose()
        self.residuals_normalized = (z - z_x_est) / np.sqrt(self.cov_noise[0,0])
        
    
        return self.residuals_normalized

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

        # Find all anomalous values
        with np.nditer(r, flags=["multi_index"]) as it:
            for value in it:
                if not np.isinf(value):
                    # nan values (i.e. 0/0) are handled by the fact that
                    # any comparison with a nan will return False
                    if abs(value) > self.anomaly_threshold:
                        index, timestep = it.multi_index
                        anomalous_indexes.append((index, timestep))
                    
                    
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
    net = PowerGrid(14)
    t1 = time()
    z = net.create_measurements(4, env_noise=True)

    Hs, x_ests, z_x_ests = net.estimate_state(z)

    config = {
        "net": net,
        "H": Hs[0],
        "x_est": x_ests[0],
        "z_x_est": z_x_ests[0]
    }

    z_a = np.copy(z)

    injection_vectors = [] # [(a,c), ...]

    for ts in range(z.shape[1]):
        config["H"] = Hs[ts]
        config["x_est"] = x_ests[ts]
        config["z_x_est"] = z_x_ests[ts]
        a, c = AnomalyModels.least_effort_norm_1(**config)
        injection_vectors.append((a,c))
        z_a[:, ts] += a

    t2 = time()
    print("elapsed:", t2 - t1)

    r = net.calculate_normalized_residuals(z_a, x_ests)
    print(r)
    print(net.check_for_anomalies())