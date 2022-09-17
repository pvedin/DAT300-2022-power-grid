import pandapower as pp
import pandapower.networks as ppn
import numpy as np
import gurobipy as grb
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.decomposition import FastICA
from pandas import Series

class AnomalyModels():
    """
    A collection of methods that, given a range of parameters, return an
    injection vector 'a' that is to be applied to one or more measurements:
    z_a = z + a

    Available models include:
        - least_effort: the aim is to successfully alter a specific measurement
                        such that it involves minimal "effort".
            - least_effort_norm_1: aims to minimize the total deviation from natural measurements
            - least_effort_big_m: aims to minimize the number of nodes by minimizing
                                    the number of nodes whose non-zero individual measurements
                                    are within +- M
        - targeted_attack: Similar to least_effort_big_m, but aims to create a specific
                            change in the state variables rather than in the measurements.
        - modal_decomposition: todo
        - random_matrix: todo

    Parameters are generally passed through a {name: value} dictionary;
    see the comments for each model for more information.
    """

    def least_effort_general(model_description,
                             var_decls=lambda m, cvs: {"e", m.addVar(name='example')},
                             constraints=lambda m, cvs, vs: m.addConstr(vs["a"][1] == 1),
                             objective=(lambda vs: sum(vs["y"]), grb.GRB.MINIMIZE),
                             **const_kwargs):
        """
        Contains common code for different variants of the least effort model.
        const_kwargs is expected to include as keys:
        - H, a matrix that is derived from the grid under study
        - z, generated measurements *for a single time step* that 
          are to be perturbed
        - fixed: a {index:value} dictionary of the indexes of the target
                 vector the adversary wants to change and by how much. Note that
                 when the 'targeted' key is present, the first element of the
                 (c) vector (i.e. index 0) cannot be modified.
        - a_bounds, (lower, upper) bounds for elements of a
        - c_bounds, (lower, upper) bounds for elements of c

        It may also include the following:
        - targeted, if included, changes the target from measurements to
          state variables. This means that k and delta will be used with
          the c vector instead (with constraints changed accordingly). The
          return value remains unchanged.

        Common variables include:
        - a, the attack vector
        - a_positive and a_negative, components of a used as a workaround for
          the solver not supporting abs()
        - c, the effect of the adjustment of a on the state
        Common constraints include:
        - a[k] = delta;
        - a_bounds[0] <= a[i] <= a_bounds[1] for all i in [0, H.shape(0)[
        - c_bounds[0] <= c[i] <= c_bounds[1] for all i in [0, H.shape(1)[

        Additional variables/constraints may be included as follows:
        - secure is a variable that can be added to const_kwargs which
          should contain a list of integers indicating which measurements the
          adversary cannot access.
        - var_decls is a function that when given a grb.Model and const_kwargs, 
          adds variables to it and returns them as a dictionary.
        - constraints is a function where the model, const_kwargs, and var_decls 
          are made available for the purpose of creating additional constraints
        - objective gives a function that denotes the target and type of objective.

        The function returns a vector of perturbations that can then be added to measurements of choice.
        """
        m = grb.Model(model_description)
        cvs = const_kwargs

        vs = var_decls(m, cvs)
        a = vs["a"] = [m.addVar(name=f"a{i}") for i in range(cvs["H"].shape[0])]
        c = vs["c"] = [m.addVar(name=f"c{i}") for i in range(cvs["H"].shape[1])]
        vs["a_positive"] = [m.addVar(lb=0.0, name='a_pos%d' % i) for i in range(cvs["H"].shape[0])] 
        vs["a_negative"] = [m.addVar(ub=0.0, name='a_neg%d' % i) for i in range(cvs["H"].shape[0])]

        # Workaround for the solver not supporting abs()
        for i in range(0, cvs["H"].shape[0]):
            m.addConstr(a[i] == vs["a_positive"][i] + vs["a_negative"][i])

        target = "a"
        if "targeted" in cvs: # Target the state rather than the measurement
            target = "c"
            # The first state variable is normally expected to be 0, so ignore
            # if the user wanted to change it
            if 0 in cvs["fixed"].keys():
                del cvs["fixed"][0]

        # default:  a[k] == delta
        # targeted: c[k] == delta
        for k, delta in cvs["fixed"].items():
            m.addConstr(vs[target][k] == delta)


        adef = lambda i: a[i] == (cvs["H"] @ c)[i] # Second constraint
        if "targeted" in cvs:
            I_fixed = [0] + list(sorted(cvs["fixed"].keys()))
            I_free = [i for i in range(0, cvs["H"].shape[1]) if i not in I_fixed]
            H_s = cvs["H"][:, I_free]
            H_s_transpose = np.transpose(H_s)
            B_s = H_s @ np.linalg.inv(H_s_transpose @ H_s) @ H_s_transpose - np.eye(H_s.shape[0])
            c_aux = [c[i] for i in I_fixed]
            
            adef = lambda i: np.matmul(B_s, a)[i] == (B_s @ cvs["H"][:, I_fixed] @ c_aux)[i]
        
        # default:  a = H*c
        # targeted: B_s * a = y
        for i in range(0, cvs["H"].shape[0]):
            m.addConstr(adef(i))

        def _add_Constr_for(m, lst, bounds, _from, _to):
            for i in range(_from, _to):
                m.addConstr(lst[i] >= bounds[0])
                m.addConstr(lst[i] <= bounds[1])

        # ... <= a[i] <= ...
        _add_Constr_for(m, a, cvs["a_bounds"], 0, cvs["H"].shape[0])

        # ... <= c[i] <= ...
        m.addConstr(vs["c"][0] == 0) # todo: determine whether this should stay (perhaps targeted only?)   
        _add_Constr_for(m, c, cvs["c_bounds"], 1, cvs["H"].shape[1])

        # Inaccessible nodes (optional)
        if "secure" in cvs:
            fixed = cvs["fixed"].keys() if "fixed" in cvs else []
            for si in cvs["secure"]:
                assert si not in fixed # adversary wants to affect a value they cannot
                m.addConstr(vs[target][si] == 0)

        # Additional, optional constraints
        if constraints:
            constraints(m, cvs, vs)

        m.setObjective(objective[0](vs), objective[1])
        
        m.update()
        m.optimize()

        # Extract elements of a
        try:
            least_effort_a = [v.x for v in m.getVars() if v.VarName in [f"a{i}" for i in range(cvs["H"].shape[0])]]
        except AttributeError: # Infeasible model
            return False

        # a for a single point in time
        least_effort_a = np.asarray(least_effort_a)

        # Since the gurobi solver appears to be deterministic with regards to
        # the final answer it returns, running it once for every time step 
        # would be inefficient. Hence the user is expected to add the 
        # resulting vector to measurements of their choosing.
        return least_effort_a

    def least_effort_norm_1(**const_kwargs):
        """
        See least_effort_general for the expected contents of const_kwargs.
        """
        model_description = const_kwargs.pop("description", "Least effort with norm 1")

        def var_decls(_m, _cvs):
            return {}

        def constraints(m, _cvs, vs):
            m.addConstr(sum(ab[0]-ab[1] for ab in zip(vs["a_positive"], vs["a_negative"])) >= 1e-1)

        objective = (lambda vs: sum(ab[0]-ab[1] for ab in zip(vs["a_positive"], vs["a_negative"])), grb.GRB.MINIMIZE)
        
        a = AnomalyModels.least_effort_general(
            model_description, var_decls, constraints, objective, **const_kwargs
        )

        return a

    def least_effort_big_m(**const_kwargs):
        """
        See least_effort_general for the expected contents of const_kwargs.
        In addition, an integer parameter M is expected, which gives constraints
        similar to bounds_a; the difference being that the bounds given by M is 
        also dependent on binary values, where there is one such value for 
        each measurement and the sum of which is the target of minimization.
        """

        model_description = const_kwargs.pop("description", "Least effort with big-M")

        def var_decls(m, cvs):
            return { 
                "y": [m.addVar(vtype=grb.GRB.BINARY, name='y%d' % i) for i in range(cvs["H"].shape[0])],
                }

        # -y[i]*M <= a[i] <= y[i]*M
        def constraints(m, _cvs, vs):
            for i in range(0, _cvs["H"].shape[0]):
                m.addConstr(vs["a"][i] <= vs["y"][i] * const_kwargs["M"])
                m.addConstr(vs["a"][i] >= -vs["y"][i] * const_kwargs["M"])
                
            m.addConstr(sum(vs["y"]) >= 1)
            m.addConstr(sum(ab[0]-ab[1] for ab in zip(vs["a_positive"], vs["a_negative"])) >= 1e-1)

        objective = (lambda vs: sum(vs["y"]), grb.GRB.MINIMIZE)

        least_effort_a_big_M = AnomalyModels.least_effort_general(
            model_description, var_decls, constraints, objective, **const_kwargs
        )
        return least_effort_a_big_M 

    def targeted_least_effort_norm_1(**const_kwargs):
        """
        See least_effort_general for the expected contents of const_kwargs.
        The expected keys are the same as in least_effort_norm_1.
        """
        const_kwargs["targeted"] = True
        const_kwargs["description"] = "Targeted least effort with norm 1"
        return AnomalyModels.least_effort_norm_1(**const_kwargs)

    def targeted_least_effort_big_m(**const_kwargs):
        """
        See least_effort_big_m for the expected contents of const_kwargs.
        """
        const_kwargs["targeted"] = True
        const_kwargs["description"] = "Targeted least effort with big-M"
        return AnomalyModels.least_effort_big_m(**const_kwargs)  

    def targeted_matching_pursuit(**const_kwargs):
        """
        Implemented using the Orthogonal Matching Pursuit algorithm.
        Required keys in const_kwargs:
        - H: a matrix derived from the grid under study
        - fixed: a {index:value} dictionary of the components of the state
                 vector c the adversary wants to change and by how much. Note that
                 the first state variable (index 0) cannot be modified.

        Note that this 
        """
        cvs = const_kwargs
        
        I = list(range(0, cvs["H"].shape[1]))
        I_fixed = [0] + [i for i in sorted(cvs["fixed"].keys()) if i != 0]
        I_free = [i for i in I if i not in I_fixed]

        H_s = cvs["H"][:, I_free]
        H_s_transpose = np.transpose(H_s)
        B_s = H_s @ np.linalg.inv(H_s_transpose @ H_s) @ H_s_transpose - np.eye(H_s.shape[0])
        c = np.zeros((cvs["H"].shape[1],))
        for i in I_fixed[1:]:
            c[i] = cvs["fixed"][i]
        y = B_s @ cvs["H"][:, I_fixed] @ c[I_fixed]

        H_t = np.transpose(cvs["H"][:, 1:])
        possible_as = [] # [(# of non-zero elements, a)]
        for n in range(1, cvs["H"].shape[0] + 1):
            # Bs*a=y, solve for a where cardinality(a) = n_nonzero_coefs
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n, normalize=False)
            omp.fit(B_s, y)
            a = omp.coef_
            # a = H*c => c = (H^T * H)^(-1) * H^T * a
            resulting_c = np.linalg.inv(H_t @ cvs["H"][:, 1:]) @ H_t @ a
            resulting_c = np.hstack((0, resulting_c))
            #print((c[I_fixed] - resulting_c[I_fixed]).reshape(-1, 1))
            if np.abs(np.max(c[I_fixed] - resulting_c[I_fixed])) < 1e-4:
                break
            possible_as.append((n, a))

        return possible_as

    def modal_decomposition(z_t, error_tolerance=1e-4):
        """
        WIP.
        Uses Individual Component Analysis (ICA) to attempt to create an
        injection vector. Only measurements for a single time step is used;
        hence, the return value is not the injection vector itself but rather
        the (now perturbed) input measurements.

        If the attack is deemed infeasible, False is returned.

        Required parameters:
        z_t: measurements for a single time step, as a column vector (e.g. z[:, 1])
        """
        z = z_t.reshape((-1, 1))
        ica = FastICA(n_components=None, random_state=0,whiten='unit-variance', fun='logcosh')
        y = ica.fit_transform(z)
        G = ica.mixing_

        # Check for feasibility
        assert np.max(np.abs(z - (np.dot(y, G.T) + ica.mean_))) > error_tolerance

        sigma_2 = 0.00001 # to be revisited
        mean = np.zeros(y.shape[0])
        cov = np.eye(y.shape[0]) * sigma_2
        delta_y = np.random.multivariate_normal(mean, cov).reshape((-1, 1))
        z_perturbed = z + np.dot(y + delta_y, G.T) + ica.mean_

        return z_perturbed

    def random_matrix(**cvs):
        """
        Attempts to compute an injection vector a that aims to tamper with a
        specific state variable, with reduced knowledge about the grid
        (i.e. the adversary only has access to the dimensions of H).

        Required parameters:
        - H, a matrix that is derived from the grid under study.
             Note that only the dimensions of H are used.
        - T, the max size of the rolling window (e.g. 4*cvs["H"].shape[1])
        - Z, a matrix of measurements for n time stepssuch that n > T
        - k, the index of the state variable of interest
        - delta, the desired perturbation
        """

        T = 40 * cvs["H"].shape[1] # Default value
        if "T" in cvs:
            T = cvs["T"] 

        # Ensure there are enough measurements
        assert cvs["t"] > T 
        assert cvs["Z"].shape[1] >= cvs["t"]

        # Grab the T most recent sets of measurements
        Z = cvs["Z"][:, cvs["t"] - T: cvs["t"]]


        Z_bar = np.mean(Z, axis=1) / (Z.shape[1] - 1)

        lst = []
        for i in range(Z.shape[1]):
            lst.append(np.outer(Z[:, i] - Z_bar, Z[:, i] - Z_bar))
        
        cov = sum(lst) / (Z.shape[1] - 1)

        u, _s, _v = np.linalg.svd(cov)
        c = np.zeros((cvs["H"].shape[1], 1))  # constant
        c[cvs["k"]] = cvs["delta"]
        
        a = u[:, :cvs["H"].shape[1]] @ c # for a single time t

        # a is a vector of single-element vectors
        return a.reshape(-1)



class PowerGrid():
    """
    Logical representation of a (preset) network, with methods for state estimation
    and emulation of anomalous data. 
    """
    # These variables are created upon initialization, and could be considered
    # constants. They are set through the given input parameters.
    network            = None # 'network_id'
    H                  = None # 'network_id'
    measurement_factor = 1/5000 # 'measurement_factor', optional
    noise_factor       = 1/500 # 'noise_factor', optional
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
                delta_load = np.random.multivariate_normal(mean, self.cov)
                p_mw.add(Series(delta_load)) 
                pp.rundcpp(self.network)
                x_t = x_temp.to_numpy().reshape((x_t.shape[0], -1))
                x_t_mat = np.hstack((x_t_mat, x_t))
        else:
            raise ValueError(f"Invalid data_generation_strategy: {data_generation_strategy}")

        self.data_generation_strategy = data_generation_strategy

        #z_t = H @ x_t + noise
        mean = np.zeros(self.H.shape[0])
        self.cov_noise = np.eye(self.H.shape[0]) * self.noise_factor
        noise_mat = np.transpose(np.random.multivariate_normal(mean, self.cov_noise, size=T))
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
        print(H_est.shape, z.shape)
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
        self.residuals_normalized = r / np.sqrt(self.cov_noise[0, 0]) # check if abs(measurement)>3
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
