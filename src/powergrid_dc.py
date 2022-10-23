print("Loading libraries...")
import pandapower as pp
import pandapower.networks as ppn
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.decomposition import FastICA
from pandas import Series
from scipy import sparse
print("Loaded!")

class AnomalyModels():
    """
    A collection of methods that, given a range of parameters, return an
    injection vector 'a' that is to be applied to one or more measurements:
    z_a = z + a

    Available models include:
        - least_effort: the aim is to successfully alter a measurement
                        such that it involves minimal "effort"
            - least_effort_norm_1: aims to minimize the total deviation from 
                                   natural measurements
            - least_effort_big_m: aims to minimize the number of nodes by minimizing
                                  the number of nodes whose non-zero individual 
                                  measurements are within +- M
        - small_ubiquitous: aim to alter a measurement such that the magnitude of
                            other needed perturbations are minimized, at the 
                            expense of requiring more such perturbations
        - targeted_*: variants of some models that aim to induce a speciic change
                      in the state vector rather than the measurements.
        - modal_decomposition: aims to perturb a measurement vector while having
                               no knowledge of the H matrix
        - random_matrix: aims to perturb the measurement vector while only 
                         knowing the dimensions of the H matrix and 
                         previous measurements

    Parameters are generally passed through a {name: value} dictionary;
    see the comments for each model for more information.
    """

    def least_effort_general(var_decls=lambda m, cvs: {},
                             constraints=lambda m, cvs: None,
                             objective=(lambda m: None, pyo.minimize),
                             **const_kwargs):
        """
        Contains common code for different variants of the least effort model.
        const_kwargs is expected to include as keys:
        - H, a matrix that is derived from the grid under study
        - fixed: a {index:value} dictionary of the indexes of the target
                 vector the adversary wants to change and by how much. Note that
                 when the 'targeted' key is present, the first element of the
                 (c) vector (i.e. index 0) cannot be modified.
        
        It may also include the following:
        - targeted, if included, changes the target from measurements to
          state variables. This means that k and alpha will be used with
          the c vector instead (with constraints changed accordingly). The
          return value continues to be the a vector.
        - a_bounds, (lower, upper) bounds for elements of a (default: [-1000, 1000]) 
        - c_bounds, (lower, upper) bounds for elements of c (default: [-1000, 1000])

        Common variables (accessible through the variable m) include:
        - a, the attack vector
        - a_pos and a_neg, components of a used as a workaround for
          the solver not supporting abs()
        - c, the effect of the adjustment of a on the state
        Common constraints include:
        - a[k] = alpha (or c[k] = alpha for targeted injections);
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
        - silence is an attribute which, if present, will stop Gurobi from printing
          to the console. This includes the "Infeasible model!" error message.

        The function returns a vector of perturbations that can then be added to 
        measurements of choice.
        A zero vector will be returned if the input model is found to be infeasible.
        """
        m = pyo.ConcreteModel()
        cvs = const_kwargs

        vs = var_decls(m, cvs)
        m.a_num = range(cvs["H"].shape[0])
        m.c_num = range(cvs["H"].shape[1])
        m.a = vs["a"] = pyo.Var(m.a_num, domain=pyo.Reals, bounds=cvs.get("a_bounds", (-1000, 1000)), initialize=0)
        m.c = vs["c"] = pyo.Var(m.c_num, domain=pyo.Reals, bounds=cvs.get("c_bounds", (-1000, 1000)), initialize=0)
        
        # Workaround for the solver not supporting abs()
        m.a_pos = vs["a_pos"] = pyo.Var(m.a_num, domain=pyo.NonNegativeReals, initialize=0)
        m.a_neg = vs["a_neg"] = pyo.Var(m.a_num, domain=pyo.NonPositiveReals, initialize=0)
        m.abs_a = pyo.ConstraintList()
        for i in m.a_num:
            m.abs_a.add(m.a[i] == m.a_pos[i] + m.a_neg[i]) 

        target = "a"
        if "targeted" in cvs: # Target the state rather than the measurement
            target = "c"
            # The first state variable is normally expected to be 0, so ignore
            # if the user wanted to change it
            if 0 in cvs["fixed"].keys():
                del cvs["fixed"][0]

        # default:  a[k] == alpha
        # targeted: c[k] == alpha
        m.targets = pyo.ConstraintList()
        for k, alpha in cvs["fixed"].items():
            m.targets.add(vs[target][k] == alpha)

        adef = lambda i: m.a[i] == (cvs["H"].todense() @ m.c)[i] # Second constraint
        if "targeted" in cvs:
            I_fixed = [0] + list(sorted(cvs["fixed"].keys()))
            I_free = [i for i in range(0, cvs["H"].shape[1]) if i not in I_fixed]
            H_s = cvs["H"][:, I_free]
            H_s_transpose = np.transpose(H_s)
            B_s = H_s @ sparse.linalg.inv(H_s_transpose @ H_s) @ H_s_transpose - np.eye(H_s.shape[0])
            A = np.eye(cvs["H"].shape[1])
            A = np.delete(A, I_free, 0)
            c_aux = A @ m.c
            mat = B_s @ cvs["H"][:, I_fixed].todense() 
            def adef(i):
                res = (B_s @ m.a)[i] == (mat @ c_aux)[i]
                return res if res is not True else pyo.Constraint.Feasible
        
        # default:  a = H*c
        # targeted: B_s * a = y
        m.injection_constraint = pyo.ConstraintList()
        for i in range(0, cvs["H"].shape[0]):
            m.injection_constraint.add(adef(i))

        m.inaccessible = pyo.ConstraintList()
        m.inaccessible.add(m.c[0] == 0)

        # Inaccessible nodes (optional)
        if "secure" in cvs:
            fixed = cvs["fixed"].keys() if "fixed" in cvs else []
            for si in cvs["secure"]:
                if si in fixed: # adversary wants to affect a value they cannot
                    return np.zeros(cvs["H"].shape[0]) # infeasible
                m.inaccessible.add(m.a[si] == 0)

        # Additional, optional constraints
        if constraints:
            constraints(m, cvs)

        m.value = pyo.Objective(rule=objective[0], sense=objective[1])

        output_flag = "silence" not in cvs
        
        # Use the Gurobi solver
        optimizer = pyo.SolverFactory("gurobi", solver_io="python")
        
        num_solutions = cvs.get("multiple_solutions", 0)
        if num_solutions:
            optimizer.options["PoolSearchMode"] = 2
            optimizer.options["PoolSolutions"] = num_solutions

        results = optimizer.solve(m, tee=output_flag)

        # Extract elements of a if successful
        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            solutions = optimizer._solver_model.SolCount
            if num_solutions and solutions > 1:
                xs = []
                for s in range(solutions):
                    optimizer._solver_model.params.SolutionNumber = s
                    x=np.asarray(optimizer._solver_model.getAttr("Xn"))
                    xs.append(x[:len(m.a_num)])
                return xs
            else:
                a_output = np.asarray([pyo.value(m.a[i]) for i in m.a_num])
                return a_output # Injection vector for a single time step
        elif (results.solver.termination_condition == TerminationCondition.infeasible):
            print("Infeasible!")
        else:
            print(f"Error! Solver status: {results.solver.status}, Termination condition: {results.termination_condition}")

        # Return zero vector if not successful
        a_zero = np.zeros(cvs["H"].shape[0])
        return a_zero

    def least_effort_norm_1(**const_kwargs):
        """
        const_kwargs is expected to contain "H" and "fixed"; see
        least_effort_general for more information.
        """

        def constraints(m, _cvs):
            m.nonzero_a = pyo.Constraint(expr = sum(m.a_pos[i] - m.a_neg[i] for i in m.a_num) >= 1e-1)

        objective = (lambda m: sum(m.a_pos[i] - m.a_neg[i] for i in m.a_num), pyo.minimize)
        
        a = AnomalyModels.least_effort_general(
            constraints = constraints, 
            objective = objective, 
            **const_kwargs
        )

        return a

    def least_effort_big_m(**const_kwargs):
        """
        const_kwargs is expected to contain "H" and "fixed"; see
        least_effort_general for more information.
        In addition, an integer parameter M can be given (default value: 1000)
        which gives constraints similar to bounds_a; the difference being that 
        the bounds given by M are also dependent on binary values, where there 
        is one such value for  each measurement and the sum of which is the 
        target of minimization.
        """
        ys = const_kwargs["H"].shape[0]

        def var_decls(m, cvs):
            vs = {}
            m.y = vs["y"] = pyo.Var(range(cvs["H"].shape[0]), domain=pyo.Binary, initialize=0)
            return vs
            
        # -y[i]*M <= a[i] <= y[i]*M
        def constraints(m, cvs):
            m.bigM_bounds = pyo.ConstraintList()
            M = cvs.get("M", 1000)
            
            for i in range(0, ys):
                m.bigM_bounds.add(m.a[i] <= m.y[i] * M)
                m.bigM_bounds.add(m.a[i] >= -m.y[i] * M)
                
            m.nonzero_y = pyo.Constraint(expr = sum(m.y[i] for i in range(ys)) >= 1)
            m.nonzero_a = pyo.Constraint(expr = sum(m.a_pos[i] - m.a_neg[i] for i in m.a_num) >= 1e-1)

        objective = (lambda m: sum(m.y[i] for i in range(ys)), pyo.minimize)

        least_effort_a_big_M = AnomalyModels.least_effort_general(
            var_decls, constraints, objective, **const_kwargs
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
        """
        cvs = const_kwargs
        
        I = list(range(0, cvs["H"].shape[1]))
        I_fixed = [0] + [i for i in sorted(cvs["fixed"].keys()) if i != 0]
        I_free = [i for i in I if i not in I_fixed]

        H_s = cvs["H"][:, I_free].todense()
        H_s_transpose = np.transpose(H_s)
        B_s = H_s @ np.linalg.inv(H_s_transpose @ H_s) @ H_s_transpose - np.eye(H_s.shape[0])
        c = np.zeros((cvs["H"].shape[1],))
        for i in I_fixed[1:]:
            c[i] = cvs["fixed"][i]
        y = B_s @ cvs["H"][:, I_fixed] @ c[I_fixed]

        H_t = np.transpose(cvs["H"][:, 1:]).todense()
        for n in range(1, cvs["H"].shape[0] + 1):
            # Bs*a=y, solve for a where cardinality(a) = n_nonzero_coefs
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n, normalize=False)
            y = np.asarray(list(y)).flatten()
            omp.fit(B_s, y)
            a = omp.coef_
            # a = H*c => c = (H^T * H)^(-1) * H^T * a
            resulting_c = np.linalg.inv(H_t @ cvs["H"][:, 1:]) @ H_t @ a
            resulting_c = np.asarray(list(resulting_c)).flatten()
            resulting_c = np.hstack((0, resulting_c.reshape(-1)))

            if np.max(np.abs(c[I_fixed] - resulting_c[I_fixed])) < 1e-4:
                return a
                
        # Infeasible
        return np.zeros((cvs["H"].shape[0],))

    def small_ubiquitous(**const_kwargs):
        """
        See least_effort_general for the expected contents of const_kwargs.
        """
        # Minimize norm-2 (note that gurobi does not support square roots)
        objective = (lambda m: sum(map(lambda x: x**2, (m.a[i] for i in m.a_num))), pyo.minimize)
        
        a = AnomalyModels.least_effort_general(
            objective = objective, 
            **const_kwargs
        )

        return a

    def targeted_small_ubiquitous(**cvs):
        """
        Required keys in cvs:
        - H: a matrix derived from the grid under study
        - fixed: a {index:value} dictionary of the components of the state
                 vector c the adversary wants to change and by how much. Note that
                 the first state variable (index 0) cannot be modified.

        Note that singular value decomposition is used instead of the solver
        as it is more efficient.
        """
        
        I_fixed = [0] + list(sorted(cvs["fixed"].keys()))
        I_free = [i for i in range(0, cvs["H"].shape[1]) if i not in I_fixed]
        H_s = cvs["H"][:, I_free]
        H_s_transpose = np.transpose(H_s)
        B_s = H_s @ sparse.linalg.inv(H_s_transpose @ H_s) @ H_s_transpose - np.eye(H_s.shape[0])
        c = np.zeros((cvs["H"].shape[1],))
        for i in I_fixed[1:]:
            c[i] = cvs["fixed"][i]
        y = B_s @ cvs["H"][:, I_fixed] @ c[I_fixed]

        bsr = np.linalg.matrix_rank(B_s)
        try:
            u,s,v = sparse.linalg.svds(B_s, k=bsr)
        except np.linalg.LinAlgError: # e.g. did not converge
            return np.zeros(cvs["H"].shape[0])
        a = np.transpose(v) @ sparse.linalg.inv(sparse.diags(s)) @ np.transpose(u) @ np.transpose(y)
        b = np.asarray(list(a)).flatten() # (20,1) => (20,)
        return b

    def modal_decomposition(**cvs):
        """
        Uses Individual Component Analysis (ICA) to attempt to create an
        injection vector. Only measurements for a single time step is used;
        hence, the return value is not the injection vector itself but rather
        the (now perturbed) input measurements.

        Note that the injected values are sampled from a zero-mean normal 
        distribution with standard deviation 1.

        If the attack is deemed infeasible, a zero-vector is returned.

        Required parameters:
        t: the timestamp of the measurement for which a perturbation is
             desired.
        Z: a matrix of measurements for n time steps such that n >= t.
             Measurements for time steps >= t will be ignored.

        Optional parameters:
        T: the max size of the rolling window. By default, this is
           defined as z[0].shape[0] (== H.shape[0]).
        n: n_components (default: find lowest iteratively)
        f: ICA function to use (one of ["logcosh", "exp", "cube"]; default: find best iteratively)
        sigma: determines the size of the covariance matrix 
               (and hence the injection noise). The default value is 1.
        error_tolerance: threshold for when the model is considered to be applicable.
                         The default value is 1e-4.
        """
        error_tolerance = cvs.get("error_tolerance", 1e-4)
        T = cvs.get("T", 2 * cvs["Z"][0].shape[0])
        t = cvs["t"]
        Z = cvs["Z"][:, t - T:t] 
        sigma_2 = cvs.get("sigma", 1)

        # Load parameters (default: determine best)
        nr = cvs.get("n", range(1, Z.shape[0]+1))
        if type(nr) == type(1) or nr is None:
            nr = [nr]
        fr = cvs.get("f", ["logcosh", "exp", "cube"])
        if type(fr) == type("str"):
            fr = [fr]

        min_deviation = [-1, -1, 1e100] # (G, y, deviation) 

        for n in nr:
            for f in fr:
                ica = FastICA(n_components=n, random_state=0, whiten='unit-variance', fun=f)
                y = ica.fit_transform(np.transpose(Z)) # fit_transform takes as argument (n_samples, n_features))
                G = ica.mixing_
                deviation = np.max(np.abs(Z - (G @ y.T) - ica.mean_.reshape((-1,1))))
                if deviation < min_deviation[2]:
                    min_deviation = [G, y, deviation]

        if min_deviation[2] > error_tolerance:
            print("Infeasible!")
            return np.zeros((Z.shape[0], 1))

        G, y = min_deviation[:2]

        delta_y = np.zeros(y.shape[1])
        mean = np.zeros(delta_y.shape[0])
        cov = np.eye(delta_y.shape[0]) * sigma_2
        delta_y = np.random.multivariate_normal(mean, cov**2)

        a = G @ (y[-1,:] + delta_y)
        return a

    def random_matrix(**cvs):
        """
        Attempts to compute an injection vector a that aims to tamper with a
        specific state variable, with reduced knowledge about the grid
        (i.e. the adversary only has access to the dimensions of H).

        Required parameters:
        - H: a matrix that is derived from the grid under study.
             Note that only the dimensions of H are used.
        - t: the timestamp of the measurement for which a perturbation is
             desired.
        - Z: a matrix of measurements for n time steps such that n >= t-1 > T.
             Measurements for time steps >= t will be ignored.
        - state_noise: factor affecting noise introduced to the state vector; see PowerGrid.

        Optional parameters:
        - T, the max size of the rolling window (default: 10 * H.shape[1]).
        - tau: factor affecting the injection vector (default: 0.3)
        - scenario: parameter that determines how the injection vector is determined
                    (can be one of ["1a", "1b", "2"]; default: "2")
        """

        T = cvs.get("T", 10 * cvs["H"].shape[1]) 

        # Ensure there are enough measurements
        assert cvs["t"] > T 
        assert cvs["Z"].shape[1] >= cvs["t"] - 1 # Exclude the measurement to be tampered with

        # Fetch the rest of the arguments
        tau = cvs.get("tau", 0.3)
        scenario = str(cvs.get("scenario", "2"))
        noise = cvs["state_noise"]

        # Grab the T most recent sets of measurements
        Z = cvs["Z"][:, cvs["t"] - T: cvs["t"]]

        snapshots = Z.shape[1]

        Z_mean = np.sum(Z, axis=1) / (snapshots - 1)

        s = 0
        for i in range(snapshots):
            s += np.outer(Z[:, i] - Z_mean, Z[:, i] - Z_mean)
        
        sigma_z = s / (snapshots - 1)
        
        lambdas, eigenvectors = np.linalg.eigh(sigma_z)

        p = Z.shape[0] / snapshots # ratio of #measurements over #snapshots
        N = lambdas.shape[0]

        def mu(i):
            return (lambdas[i] + 1 - p + np.sqrt((lambdas[i] + 1 - p)**2 - 4 * lambdas[i])
                      )/2 - 1

        def omega(i, _mu=None):
            if not _mu: # optimization possible if mu is computed beforehand
                _mu = mu(i)
            return (1 - p/_mu**2)/ (1 + p/_mu)
 
        a = 0

        if scenario in ("1a", "1b"):
            if scenario == "1a": # Use the smallest eigenvalue
                i = np.argmin(lambdas)
            else: # Use the largest eigenvalue
                i = np.argmax(lambdas)
            _mu = mu(i)
            c_i = np.sqrt(tau/(noise**2 * omega(i, _mu)/_mu))
            a = eigenvectors[:,i] * c_i
        elif scenario == "2": # Default
            c = []
            N = lambdas.shape[0]
            for i in range(N):
                _mu = mu(i)
                c.append(np.sqrt(tau/(noise**2 * N*omega(i, _mu)/_mu)))
            c = np.asarray(c)
            a = eigenvectors @ c

        # a is a vector of single-element vectors
        return a.reshape(-1)


class PowerGrid():
    """
    Logical representation of a (preset) network, with methods for state estimation
    and emulation of anomalous data. 
    """
    # These variables are created upon initialization, and could be considered
    # constants. They are set through the given input parameters.
    network                  = None # 'network_id'
    H                        = None # 'network_id'
    state_noise_factor       = 1e-3 # 'state_noise_factor', optional
    measurement_noise_factor = 1e-2 # 'measurement_noise_factor', optional
    anomaly_threshold        = 3    # 'anomaly_threshold', optional
    
    # These variables are updated (and overwritten) upon calling the given
    # functions.
    data_generation_strategy = None # create_measurements()
    z_buffer                 = None # create_measurements()
    cov                      = None # create_measurements(); used when generating noise
    cov_noise                = None # create_measurements(); used when generating noise and normalizing residuals
    x_est                    = None # estimate_state()
    residuals_normalized     = None # calculate_normalized_residuals()

    def __init__(self, network_id, state_noise_factor=None, measurement_noise_factor=None, 
                 anomaly_threshold=None, double_measurements=False):
        """
        network_id can have the format "{prefix}{number}", {number} or "{number}",
        where prefix is one of ["IEEE ", "Illinois ", "PEGASE ", "RTE "].
        Since there does not appear to be any overlap between test cases, the 
        prefix has no functional purpose (aside from giving more context to the user).
        The number (corresponding to the number of nodes) can be one of:
        [4, 5, 6, 9, 11, 14, 24, 29, 30, 33, 39, 57, 89, 118, 145,
         200, 300, 1354, 2224, 2848, 3120, 6470, 6495, 9515, 9241].

        Some cases have special names, including as "iceland" (118 nodes, but
        different from IEEE 118), "GBnetwork" (2224 nodes) and 
        "GBreducednetwork" (29 nodes). Moreover, "IEEE30" and "IEEE 30" are
        based on the same case, but are slightly different due to having
        different origins. The former is only chosen if the network_id is
        exactly "IEEE30".

        The selected network will be loaded and stored in self.network.
        """

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
        pp.rundcpp(self.network)

        # P_from_node_i_to_node_j
        # Pij = (1/bij)*(x[i]-x[j])
        # H[line_id,i] = 1/bij
        # H[line_id,j] = -1/bij
        A_real = np.real(self.network._ppc['internal']['Bbus'].A)
        line = self.network.line
        rows = line.shape[0]
        connections = rows + self.network.trafo.shape[0]
        step = 1 + int(double_measurements)
        self.H = np.zeros((step*connections, self.network.bus.shape[0]))

        # Some connections are found in {from, to}_bus, others are found in
        # trafo.{hv, lv}_bus.
        
        for from_index, to_index, from_bus, to_bus, j in (
            (0, rows, 
                line.from_bus.values, line.to_bus.values, 
                lambda i: i), 
            (rows, connections, 
                self.network.trafo.hv_bus.values, self.network.trafo.lv_bus.values,
                lambda i: i - rows)
            ):
            for i in range(from_index, to_index):
                from_j = from_bus[j(i)]
                to_j = to_bus[j(i)]
                power_flow = A_real[from_j, to_j]
                # from flows
                self.H[step*i, from_j] = -power_flow
                self.H[step*i, to_j]  = power_flow
                # to flows
                if double_measurements:
                    power_flow = A_real[to_j, from_j]
                    self.H[step*i+1, to_j] = -power_flow
                    self.H[step*i+1, from_j]  = power_flow

        self.H = sparse.csr_matrix(self.H)

        if measurement_noise_factor:
            self.measurement_noise_factor = measurement_noise_factor
        if state_noise_factor:
            self.state_noise_factor = state_noise_factor
        if anomaly_threshold:
            self.anomaly_threshold = anomaly_threshold
        

    def create_measurements(self, T, data_generation_strategy, env_noise=True, OU=None):
        """
        Generates T measurements according to
            z = H*x + n
        where H is derived from a preset network configuration, x is generated
        based on the chosen data_generation_strategy, and n represents random
        noise.

        The argument data_generation_strategy can be either 1 or 2.

        env_noise will omit the noise added to each measurement if set to
        False; however, the noise added to the state vectors remains unchanged.

        Returns a [X x T] numpy array -- where X is the number of measurements
        for each time step -- which is also stored in self.z_buffer.
        """
        x_temp = self.network.res_bus.va_degree * (np.pi / 180) # Store in radians
        if data_generation_strategy == 1:
            # state vector x_base
            x_base = x_temp.to_numpy()
            # Dx ~ N(0,Cov_Mat)
            mean = np.zeros(x_base.shape)
            self.cov = np.eye(x_base.shape[0]) * self.state_noise_factor
            delta_x_mat = np.transpose(np.random.multivariate_normal(mean, self.cov**2, size=T))
            x_base_mat = np.repeat(x_base.reshape((x_base.shape[0], -1)), T, axis=1)
            x_t_mat = x_base_mat + delta_x_mat
            x_t_mat[0, :] = 0

        elif data_generation_strategy == 2:
            """
            Jesper Franke, "Short introduction to python", Potsdam Institute for Climate Impact Research
            https://www.pik-potsdam.de/members/franke/lecture-sose-2016/introduction-to-python.pdf

            Experimental WIP. Has not been thoroughly tested.
            """
            
            x_t = x_temp.to_numpy()
            x_t_mat = x_t.reshape((x_t.shape[0], -1))
            p_mw_init = self.network.load.p_mw.values.copy()

            if OU:
                theta, mu, sigma = OU
            else:
                # Comments based on article by Diego Barba (May 3, 2022)
                # https://towardsdatascience.com/stochastic-processes-simulation-the-ornstein-uhlenbeck-process-e8bff820f3
                theta = np.abs(np.random.normal(0.01, 0.1)) # mean-reversion
                mu = 0                                      # asymptotic mean
                sigma = np.abs(np.random.normal(0.04, 0.1)) # random scale

            t = np.linspace(0, T, T)
            dt = np.mean(np.diff(t))
            y = np.zeros(T)
            y[0] = 0.1

            drift = lambda y,t: theta * (mu - y)
            diffusion = lambda _y, _t: sigma
            noise = np.random.normal(0, 1, T) * np.sqrt(dt)
            #noise = np.random.normal(0, self.state_noise_factor**2, T) * np.sqrt(dt)

            for i in range(1, T):
                y[i] = y[i-1] + drift(y[i-1], i*dt)*dt + diffusion(y[i-1], i*dt) * noise[i]

            y = np.asarray(y)
            x_t_mat = []
            for t in range(T):
                delta_load = p_mw_init * (1 + y[t])
                self.network.load.p_mw = Series(delta_load)
                pp.rundcpp(self.network)
                x_t_mat.append(self.network.res_bus.va_degree * (np.pi / 180)) # Store in radians

            x_t_mat = np.transpose(np.asarray(x_t_mat))

            self.network.load.p_mw = p_mw_init

            #mean = np.zeros(p_mw.shape) * self.state_noise_factor
            #self.cov = np.eye(p_mw.shape[0])    
            #for t in range(1, T):
            #    delta_load = np.random.multivariate_normal(mean, self.cov**2)
            #    p_mw.add(Series(delta_load)) 
            #    pp.rundcpp(self.network)
            #    x_t = x_temp.to_numpy().reshape((x_t.shape[0], -1))
            #    x_t_mat = np.hstack((x_t_mat, x_t))
        else:
            raise ValueError(f"Invalid data_generation_strategy: {data_generation_strategy}")

        self.data_generation_strategy = data_generation_strategy

        #z_t = H @ x_t + noise, where @ is infix for np.matmul
        mean = np.zeros(self.H.shape[0])
        self.cov_noise = np.eye(self.H.shape[0]) * self.measurement_noise_factor
        noise_mat = np.transpose(np.random.multivariate_normal(mean, self.cov_noise**2, size=T))
        self.W = np.linalg.inv(self.cov_noise**2) # Used in estimate_state()
        z_t_mat = self.H @ x_t_mat
        if env_noise:
            self.z_buffer = z_t_mat + noise_mat
        return z_t_mat

    def estimate_state(self, z=None):
        """
        Calculates state estimations based on the given network configuration
        and observed measurements according to:
            x_est = (H_est_transpose * W * H_est)^-1 * H_est_transpose * z
        where W[i,i] = 1/sigma**2 and H_est is equal to H with the exception that the first column
        is removed (under the assumption that the remaining state variables
        x[1:, :] are relative to x[0, :]).
        """
        z = self._load_var("z_buffer", z, err_msg="Cannot estimate state without measurements.")

        H_est = self.H[:, 1:].copy()
        H_est_transpose = np.transpose(H_est)

        # If H was a numpy array:
        # self.x_est = np.linalg.inv(H_est_transpose @ self.W @ H_est) @ H_est_transpose @ self.W @ z
        self.x_est = sparse.linalg.spsolve(H_est_transpose @ self.W @ H_est, H_est_transpose @ self.W @ z)

        if len(self.x_est.shape) == 1: # e.g. (13,) instead of (13, 1)
            self.x_est = self.x_est.reshape(-1, 1)

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

        # If e ~ N(0, self.cov_noise**2), then normalized_residuals ~ N(0, omega)
        Ht = np.transpose(self.H[:,1:])
        omega = self.cov_noise**2 - (self.H[:,1:] @ np.linalg.inv(Ht @ self.W @ self.H[:,1:]) @ Ht)
        omega[abs(omega) < 1e-8] = 1e-8
        self.residuals_normalized = []

        for i in range(0, r.shape[1]):
            v = np.abs(r[:,i]) / np.sqrt(np.abs(np.diag(omega)))
            self.residuals_normalized.append(v)
        
        self.residuals_normalized = np.transpose(np.asarray(self.residuals_normalized))

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
