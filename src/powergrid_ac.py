print("Loading libraries...")
import pandapower as pp
import pandapower.networks as ppn
from pandapower.pypower.dSbr_dV import dSbr_dV
from scipy import sparse
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import idaes
from time import time

print("Loaded!")

class AnomalyModels():

    def least_effort_norm_1(**const_kwargs):
        """
        const_kwargs is expected to include as keys:
        - H, a matrix that is derived from the grid under study
        - z, generated measurements *for a single time step* that 
          are to be perturbed
        - x_est, the estimated state for the given z
        - z_x_est, the estimated measurement for the given x_est (i.e. h(x_est))
        - fixed: a {index:value} dictionary of the indexes of the measurement
                 vector the adversary wants to change and by how much.

        Optional variables may be included:
        - a_bounds, (lower, upper) bounds for elements of a
        - c_bounds, (lower, upper) bounds for elements of c
        In both of the above, the default bounds are [-1000, 1000].

        Additional variables/constraints may be included as follows:
        - secure is a variable that can be added to const_kwargs which
          should contain a list of integers indicating which measurements the
          adversary cannot access.
        - silence is an attribute which, if present, will stop Gurobi from printing
          to the console. This includes the "Infeasible model!" error message.
        - constraints is a function f(m, cvs) that can be used to add additional
          constraints.

        The function returns a vector of perturbations that can then be added to 
        measurements of choice. The resulting perturbation to c is also returned.
        A zero vector will be returned if the input model is found to be infeasible.
        """
        cvs = const_kwargs
        net = cvs["net"].network
        H = cvs["H"]
        x_est = cvs["x_est"]
        z_x_est = cvs["z_x_est"].reshape(-1)

        m = pyo.ConcreteModel()

        m.a_num = range(H.shape[0])
        m.c_num = range(H.shape[1]+1)
        m.a = pyo.Var(m.a_num, domain=pyo.Reals, bounds=cvs.get("a_bounds", (-1000, 1000)), initialize=0)
        m.c = pyo.Var(m.c_num, domain=pyo.Reals, bounds=cvs.get("c_bounds", (-1000, 1000)), initialize=0)
        m.a_pos = pyo.Var(m.a_num, domain=pyo.NonNegativeReals, initialize=0)
        m.a_neg = pyo.Var(m.a_num, domain=pyo.NonPositiveReals, initialize=0)

        m.abs_a = pyo.ConstraintList()
        for i in m.a_num:
            m.abs_a.add(m.a[i] == m.a_pos[i] + m.a_neg[i]) 
            
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
                        - x_est_m[bi0] * x_est_m[bi1]
                        * (g[i] * pyo.cos(x_est_a[bi0] - x_est_a[bi1])
                           + b[i] * pyo.sin(x_est_a[bi0] - x_est_a[bi1])) 
                        / tap[i])

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

        m.change_a = pyo.ConstraintList()
        for k, alpha in cvs["fixed"].items():
            m.change_a.add(m.a[k] == alpha) # in reality, a change by 0.1 is 0.1*100 MW = 10 MW

        a_zero = np.zeros(H.shape[0])
        c_zero = np.zeros(H.shape[1])

        m.inaccessible = pyo.ConstraintList()
        m.inaccessible.add(m.c[0] == 0)

        # Inaccessible nodes (optional)
        if "secure" in cvs:
            fixed = cvs["fixed"].keys() if "fixed" in cvs else []
            for si in cvs["secure"]:
                if si in fixed: # adversary wants to affect a value they cannot
                    return a_zero, c_zero # infeasible
                m.inaccessible.add(m.a[si] == 0)

        # Additional, optional constraints
        if "constraints" in cvs:
            cvs["constraints"](m, cvs)
            
        def abs_a_val(m):
            return sum(m.a_pos[i] - m.a_neg[i] for i in m.a_num)
            
        m.value = pyo.Objective(rule = abs_a_val, sense = pyo.minimize)    

        output_flag = "silence" not in cvs

        optimizer = pyo.SolverFactory('ipopt')
        results = optimizer.solve(m, tee=output_flag)

        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            a_output = np.asarray([pyo.value(m.a[i]) for i in m.a_num])
            c_output = np.asarray([pyo.value(m.c[i]) for i in m.c_num])
            return a_output, c_output
        elif (results.solver.termination_condition == TerminationCondition.infeasible):
            print("Infeasible!")
        else:
            print(f"Error! Solver status: {results.solver.status}, Termination condition: {results.termination_condition}")

        return a_zero, c_zero



class PowerGrid():
    """
    Logical representation of a (preset) network, with methods for state estimation
    and emulation of anomalous data. 
    """
    # These variables are created upon initialization, and could be considered
    # constants. They are set through the given input parameters.
    network                  = None # 'network_id'
    state_noise_factor       = 1e-3 # 'state_noise_factor', optional
    measurement_noise_factor = 1e-3 # 'measurement_noise_factor', optional
    anomaly_threshold        = 3    # 'anomaly_threshold', optional
    
    # These variables are updated (and overwritten) upon calling the given
    # functions.
    data_generation_strategy = None # create_measurements()
    z_buffer                 = None # create_measurements()
    cov                      = None # create_measurements(); used when generating noise
    cov_noise                = None # create_measurements(); used when generating noise and normalizing residuals
    Hs                       = None # estimate_state()
    x_ests                   = None # estimate_state()
    residuals_normalized     = None # calculate_normalized_residuals()

    def __init__(self, network_id, state_noise_factor=None, measurement_noise_factor=None, 
                 anomaly_threshold=None):
        accepted_prefixes = ["IEEE ", "Illinois ", "PEGASE ", "RTE "]
        if type(network_id) == type(int()):
            network_id = str(network_id)
        else:
            prefix = list(p for p in accepted_prefixes if p in network_id)
            if prefix:
                prefix = prefix[0]
                network_id = network_id[network_id.index(prefix) + len(prefix):]
        cases = {
            "4": ppn.case4gs,
            "4gs": ppn.case4gs,
            "5": ppn.case5,
            "6": ppn.case6ww,
            "6ww": ppn.case6ww,
            "9": ppn.case9,
            "11": ppn.case11_iwamoto,
            "14": ppn.case14,
            "24": ppn.case24_ieee_rts,
            "29": ppn.GBreducednetwork,
            "GBreducednetwork": ppn.GBreducednetwork,
            "30": ppn.case30,          # These two are based on the same case, but have
            "IEEE30": ppn.case_ieee30, # different origins (PYPOWER and MATPOWER)
            "33": ppn.case33bw,
            "39": ppn.case39,
            "57": ppn.case57,
            "89": ppn.case89pegase,
            "118": ppn.case118,
            "iceland": ppn.iceland, # 118 nodes, but different from case118
            "145": ppn.case145,
            "200": ppn.case_illinois200,
            "300": ppn.case300,
            "1354": ppn.case1354pegase,
            "1888": ppn.case1888rte,
            "2224": ppn.GBnetwork,
            "GBnetwork": ppn.GBnetwork,
            "2848": ppn.case2848rte,
            "3120": ppn.case3120sp,
            "6470": ppn.case6470rte,
            "6495": ppn.case6495rte,
            "6515": ppn.case6515rte,
            "9241": ppn.case9241pegase
        }

        if not network_id in cases:
            raise NotImplementedError(f"Unsupported network configuration: {network_id}")
        
        # Fetch network configuration
        self.network = cases[network_id]()
        pp.runpp(self.network)

        if state_noise_factor:
            self.state_noise_factor = state_noise_factor
        if measurement_noise_factor:
            self.measurement_noise_factor = measurement_noise_factor
        if anomaly_threshold:
            self.anomaly_threshold = anomaly_threshold

    def _h_x(self, x):
        Sf = []
        rs = self.network._ppc['branch'].shape[0]
        for i in range(0, rs): # to be refined
            Sf.append(x[np.real(self.network._ppc['branch'][i, 0].astype(int))]
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

        The argument data_generation_strategy can at the moment only be 1.

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
            self.cov = np.eye(x_base.shape[0]) * self.state_noise_factor
            delta_x_mat = np.transpose(np.random.multivariate_normal(mean, self.cov**2, size=T))
            x_base_mat = np.repeat(x_base.reshape((x_base.shape[0],-1)), T, axis=1)
            x_t_mat = x_base_mat + delta_x_mat

        elif data_generation_strategy == 2:
            raise NotImplementedError()
        else:
            raise ValueError(f"Invalid data_generation_strategy: {data_generation_strategy}")

        self.data_generation_strategy = data_generation_strategy

        self.z_buffer = self._h_x(x_t_mat)
        # in the DC case, we had 20 measurements for the 14-node bus since z_P (20 x 1)
        # in the AC case, we have 20 from z_P, 20 from z_Q, and 1 from z_V_mag -> 41
        rs = self.network._ppc['branch'].shape[0] * 2 + 1 # including measurements from z_Q and z_V_mag
        mean = np.zeros(rs)
        self.cov_noise = np.eye(rs) * self.measurement_noise_factor
        self.W = np.linalg.inv(self.cov_noise**2) # Used in state estimation and when computing residuals

        if env_noise:
            noise_mat = np.transpose(np.random.multivariate_normal(mean, self.cov_noise**2, size=T))
            self.z_buffer += noise_mat
        
        return self.z_buffer

    def estimate_state(self, z=None, restarts=40, max_iter=50):
        """
        Calculates state estimations based on the given network configuration.
        The optional parameters restarts and max_iter can be increased to
        increase the probability for the state estimation to converge to a
        good value (at the expense of taking more time).

        Returns lists of z_x_est, x_est and Hs; the length of each corresponding
        to the number of time steps in z.

        Aspects such as the use of dSbr_dV were inspired by pandapower's implementation
        of state estimation.
        """
        # in DC state estimator: x_est = inv(H.T @ H) @ H.T @ z
        # in DC case: x = [theta_0, theta_1, ...]
        # in AC case: x = [theta_0, theta_1, ..., theta_13, mag_0, mag_1, ..., mag_13]
        # we need theta_0 = 0

        z = self._load_var("z_buffer", z)

        self.Hs = []
        self.x_ests = []
        z_x_ests = []

        _restarts = restarts
        _max_iter = max_iter

        error_tolerance = 1e-8

        for ts in range(z.shape[1]):
            restarts = _restarts
            max_iter = _max_iter
            estimator_outcome = []
            for _ in range(restarts):
                i = 0

                # initial guess for state vector
                #x_est_angles  = np.random.uniform(-30, 30, size=len(self.network.bus)) * (np.pi / 180)
                #x_est_abs = np.random.uniform(0.8, 1.2, size=len(self.network.bus))
                #x_est = x_est_abs * np.exp(1j * x_est_angles)

                x_est = np.ones(len(self.network.bus))

                # initially, all voltage magnitudes are assumed to be 1 and all voltage angles are assumed to 0
                residuals = 1
                
                while np.max(np.abs(residuals)) > error_tolerance and i < max_iter:
                    i += 1

                    # h(x_est) for the current x_est (which is just x_hat), equivalent to H*x_hat
                    z_x_est = self._h_x(x_est)
                    
                    residuals = z[:, ts].reshape((-1,1)) - z_x_est


                    dSf_dVa, dSf_dVm, _, _, _, _ = dSbr_dV(self.network._ppc['branch'], 
                                                        self.network._ppc['internal']['Yf'], 
                                                        self.network._ppc['internal']['Yt'], 
                                                        x_est)
                    dV_dVa = 0 * np.diag(x_est)
                    dV_dVa = sparse.csr_matrix(dV_dVa[0,:])
                    dV_dVm = np.eye(x_est.shape[0])
                    dV_dVm = sparse.csr_matrix(dV_dVm[0,:])
                    
                    H_P = sparse.hstack((np.real(dSf_dVa), np.real(dSf_dVm)))
                    H_Q = sparse.hstack((np.imag(dSf_dVa), np.imag(dSf_dVm)))
                    H_V = sparse.hstack((dV_dVa, dV_dVm))
                    H = sparse.vstack((H_P, H_Q))
                    H = sparse.vstack((H, H_V))
                    H = sparse.csr_matrix(H)[:,1:]
                    Ht = np.transpose(H)
                    
                    srs = self.network._ppc['bus'].shape[0]
                    delta_x_est = sparse.linalg.spsolve(Ht @ self.W @ H, np.transpose(H) @ self.W @ residuals)
                    delta_x_est_a = delta_x_est[0:srs-1]
                    delta_x_est_m = delta_x_est[srs-1:]

                    x_est_a = np.angle(x_est[1:]) + delta_x_est_a
                    x_est_a = np.hstack((0,x_est_a))
                    x_est_m = np.abs(x_est) + delta_x_est_m
                    x_est = x_est_m * np.exp(1j * x_est_a)

                    # Early exit if converging at a local minimum > error_tolerance
                    if np.linalg.norm(np.hstack((delta_x_est_a, delta_x_est_m)), ord=2) < error_tolerance:
                        break

                res = np.max(np.abs(residuals))
                if not estimator_outcome or (estimator_outcome and res < estimator_outcome[1]):
                    estimator_outcome = [i, np.max(np.abs(residuals)), H, x_est, z_x_est]
                
                # Early exit
                if res < error_tolerance:
                    break

            _, min_res, H, x_est, z_x_est = estimator_outcome

            print("min_res", min_res)
            self.Hs.append(H)
            self.x_ests.append(x_est)
            z_x_ests.append(z_x_est)
        
        return z_x_ests, self.x_ests, self.Hs

    def calculate_normalized_residuals(self, z=None, x_ests=None, Hs=None):
        """
            Calculates residual vectors, which represent the 
            difference between observed measurements and estimated measurements, 
            according to
                r = z - h(x_est).
            for each time step (= column) in z.
            These residuals are then normalized and returned.
        """
        z = self._load_var("z_buffer", z)
        x_ests = self._load_var("x_ests", x_ests)
        Hs = self._load_var("Hs", Hs)

        self.residuals_normalized = []
        for ts in range(z.shape[1]):
            H = Hs[ts]
            Ht = np.transpose(H[:, 1:]) 
            z_x_est = self._h_x(x_ests[ts]).reshape(-1)
            r = (z[:, ts] - z_x_est)
            omega = self.cov_noise**2 - (H[:, 1:] @ np.linalg.inv(Ht @ self.W @ H[:, 1:]) @ Ht)
            
            # Too small values indicate leverage measurements, which result in 
            # infinite residuals (which should not be interpreted as anomalous)
            omega[np.abs(omega) < 1e-10] = np.nan

            v = np.abs(r) / np.sqrt(np.abs(np.diag(omega)))
            self.residuals_normalized.append(v)

        self.residuals_normalized = np.asarray(self.residuals_normalized).transpose()
    
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
            - otherwise, 'arg' is returned and the attribute is set to this value.
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
    pass