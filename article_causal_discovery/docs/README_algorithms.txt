per ogni grafo (~11000)
    per ogni dataset (10)
        per ogni algoritmo
            applica l'algo al dataset


Algorithms
    https://pypi.org/project/gcastle/
    https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle



PC

    Parameters
    ----------
    variant : str
        A variant of PC-algorithm, one of [`original`, `stable`, `parallel`].
    alpha: float, default 0.05
        Significance level.
    ci_test : str, callable
        ci_test method, if str, must be one of [`fisherz`, `g2`, `chi2`]
        See more: `castle.common.independence_tests.CITest`
    priori_knowledge: PrioriKnowledge
        a class object PrioriKnowledge


ANMNonlinear

    Parameters
    ----------
    alpha : float, default 0.05
        significance level be used to compute threshold


DirectLiNGAM

    Parameters
    ----------
    prior_knowledge : array-like, shape (n_features, n_features), optional (default=None)
        Prior knowledge used for causal discovery, where ``n_features`` is the number of features.

        The elements of prior knowledge matrix are defined as follows [1]_:

        * ``0`` : :math:`x_i` does not have a directed path to :math:`x_j`
        * ``1`` : :math:`x_i` has a directed path to :math:`x_j`
        * ``-1`` : No prior knowledge is available to know if either of the two cases above (0 or 1) is true.
    measure : {'pwling', 'kernel'}, default='pwling'
        Measure to evaluate independence: 'pwling' [2]_ or 'kernel' [1]_.
    thresh : float,  default='0.3'
        Drop edge if |weight| < threshold


ICALiNGAM

    Parameters
    ----------
    random_state : int, optional (default=None)
        ``random_state`` is the seed used by the random number generator.
    max_iter : int, optional (default=1000)
        The maximum number of iterations of FastICA.
    thresh : float,  default='0.3'
        Drop edge if |weight| < threshold


GES

    Parameters
    ----------
    criterion: str for DecomposableScore object
        scoring criterion, one of ['bic', 'bdeu'].

        Notes:
            1. 'bdeu' just for discrete variable.
            2. if you want to customize criterion, you must create a class
            and inherit the base class `DecomposableScore` in module
            `ges.score.local_scores`
    method: str
        effective when `criterion='bic'`, one of ['r2', 'scatter'].
    k: float, default: 0.001
        structure prior, effective when `criterion='bdeu'`.
    N: int, default: 10
        prior equivalent sample size, effective when `criterion='bdeu'`


Notears

    Parameters
    ----------
    lambda1: float
        l1 penalty parameter
    loss_type: str
        l2, logistic, poisson
    max_iter: int
        max num of dual ascent steps
    h_tol: float
        exit if |h(w_est)| <= htol
    rho_max: float
        exit if rho >= rho_max
    w_threshold: float
        drop edge if |weight| < threshold


NotearsLowRank

    Parameters
    ----------
    w_init: None or numpy.ndarray
        Initialized weight matrix
    max_iter: int
        Maximum number of iterations
    h_tol: float
        exit if |h(w)| <= h_tol
    rho_max: float
        maximum for rho
    w_threshold : float,  default='0.3'
        Drop edge if |weight| < threshold

    PROBLEM:
        the method learner requires a parameter 'rank' specified by the user.
        algo.leaner(x, rank=<low_rank>)


TTPM

    Parameters
    ----------
    topology_matrix: np.matrix
        Interpreted as an adjacency matrix to generate the graph.
        It should have two dimensions, and should be square.
    delta: float, default=0.1
            Time decaying coefficient for the exponential kernel.
    epsilon: int, default=1
        BIC penalty coefficient.
    max_hop: positive int, default=6
        The maximum considered hops in the topology,
        when ``max_hop=0``, it is divided by nodes, regardless of topology.
    penalty: str, default=BIC
        Two optional values: 'BIC' or 'AIC'.
    max_iter: int
        Maximum number of iterations.

    PROBLEM: it requires 'topology_matrix'

--------------------------------------------------------------------
- Pytorch
--------------------------------------------------------------------

NontearsNonlinear

    Parameters
    ----------
    lambda1: float
        l1 penalty parameter
    lambda2: float
        l2 penalty parameter
    max_iter: int
        max num of dual ascent steps
    h_tol: float
        exit if |h(w_est)| <= htol
    rho_max: float
        exit if rho >= rho_max
    w_threshold: float
        drop edge if |weight| < threshold
    hidden_layers: Iterrable
        Dimension of per hidden layer, and the last element must be 1 as output dimension.
        At least contains 2 elements. For example: hidden_layers=(5, 10, 1), denotes two hidden
        layer has 5 and 10 dimension and output layer has 1 dimension.
        It is effective when model_type='mlp'.
    expansions: int
        expansions of each variable, it is effective when model_type='sob'.
    bias: bool
        Indicates whether to use weight deviation.
    model_type: str
        The Choice of Two Nonlinear Network Models in a Notears Framework:
        Multilayer perceptrons value is 'mlp', Basis expansions value is 'sob'.
    device_type: str, default: cpu
        ``cpu`` or ``gpu``
    device_ids: int or str, default None
        CUDA devices, it's effective when ``use_gpu`` is True.
        For single-device modules, ``device_ids`` can be int or str, e.g. 0 or '0',
        For multi-device modules, ``device_ids`` must be str, format like '0, 1'.


GOLEM

    Paramaters
    ----------
    B_init: None
        File of weighted matrix for initialization. Set to None to disable.
    lambda_1: float
        Coefficient of L1 penalty.
    lambda_2: float
        Coefficient of DAG penalty.
    equal_variances: bool
        Assume equal noise variances for likelibood objective.
    non_equal_variances: bool
        Assume non-equal noise variances for likelibood objective.
    learning_rate: float
        Learning rate of Adam optimizer.
    num_iter: float
        Number of iterations for training.
    checkpoint_iter: int
        Number of iterations between each checkpoint. Set to None to disable.
    seed: int
        Random seed.
    graph_thres: float
        Threshold for weighted matrix.
    device_type: bool
        whether to use GPU or not
    device_ids: int
        choose which gpu to use

PNL

    Parameters
    ----------
    hidden_layers: int
        number of hidden layer of mlp
    hidden_units: int
        number of unit of per hidden layer
    batch_size: int
        size of training batch
    epochs: int
        training times on all samples
    lr: float
        learning rate
    alpha: float
        significance level
    bias: bool
        whether use bias
    activation: callable
        nonlinear activation function
    device_type: str
        'cpu' or 'gpu', default: 'cpu'
    device_ids: int or str
        e.g. 0 or '0,1', denotes which gpu that you want to use.

RL

    Parameters
    ----------
    encoder_type: str
        type of encoder used
    hidden_dim: int
        actor LSTM num_neurons
    num_heads: int
        actor input embedding
    num_stacks: int
        actor LSTM num_neurons
    residual: bool
        whether to use residual for gat encoder
    decoder_type: str
        type of decoder used
    decoder_activation: str
        activation for decoder
    decoder_hidden_dim: int
        hidden dimension for decoder
    use_bias: bool
        Whether to add bias term when calculating decoder logits
    use_bias_constant: bool
        Whether to add bias term as CONSTANT when calculating decoder logits
    bias_initial_value: float
        Initial value for bias term when calculating decoder logits
    batch_size: int
        batch size for training
    input_dimension: int
        dimension of reshaped vector
    normalize: bool
        whether the inputdata shall be normalized
    transpose: bool
        whether the true graph needs transposed
    score_type: str
        score functions
    reg_type: str
        regressor type (in combination wth score_type)
    lambda_iter_num: int
        how often to update lambdas
    lambda_flag_default: bool
        with set lambda parameters; true with default strategy and ignore input bounds
    score_bd_tight: bool
        if bound is tight, then simply use a fixed value, rather than the adaptive one
    lambda1_update: float
        increasing additive lambda1
    lambda2_update: float
        increasing  multiplying lambda2
    score_lower: float
        lower bound on lambda1
    score_upper: float
        upper bound on lambda1
    lambda2_lower: float
        lower bound on lambda2
    lambda2_upper: float
        upper bound on lambda2
    seed: int
        seed
    nb_epoch: int
        nb epoch
    lr1_start: float
        actor learning rate
    lr1_decay_step: int
        lr1 decay step
    lr1_decay_rate: float
        lr1 decay rate
    alpha: float
        update factor moving average baseline
    init_baseline: float
        initial baseline - REINFORCE
    temperature: float
        pointer_net initial temperature
    C: float
        pointer_net tan clipping
    l1_graph_reg: float
        L1 graph regularization to encourage sparsity
    inference_mode: bool
        switch to inference mode when model is trained
    verbose: bool
        print detailed logging or not
    device_type: str
        whether to use GPU or not
    device_ids: int
        choose which gpu to use


GAE

    Parameters
    ----------
    input_dim: int, default: 1
        dimension of vector for x
    hidden_layers: int, default: 1
        number of hidden layers for encoder and decoder
    hidden_dim: int, default: 4
        hidden size for mlp layer
    activation: callable, default: nn.LeakyReLU(0.05)
        nonlinear functional
    epochs: int, default: 10
        Number of iterations for optimization problem
    update_freq: int, default: 3000
        Number of steps for each iteration
    init_iter: int, default: 3
        Initial iteration to disallow early stopping
    lr: float, default: 1e-3
        learning rate
    alpha: float, default: 0.0
        Lagrange multiplier
    beta: float, default: 2.0
        Multiplication to amplify rho each time
    init_rho: float, default: 1.0
        Initial value for rho
    rho_thresh: float, default: 1e30
        Threshold for rho
    gamma: float, default: 0.25
        Threshold for h
    penalty_lambda: float, default: 0.0
        L1 penalty for sparse graph. Set to 0.0 to disable
    h_thresh: float, default: 1e-8
        Tolerance of optimization problem
    graph_thresh: float, default: 0.3
        Threshold to filter out small values in graph
    early_stopping: bool, default: False
        Whether to use early stopping
    early_stopping_thresh: float, default: 1.0
        Threshold ratio for early stopping
    seed: int, default: 1230
        Reproducibility, must be int
    device_type: str, default: 'cpu'
        'cpu' or 'gpu'
    device_ids: int or str, default '0'
        CUDA devices, it's effective when ``use_gpu`` is True.
        For single-device modules, ``device_ids`` can be int or str,
        e.g. 0 or '0', For multi-device modules, ``device_ids`` must be str,
        format like '0, 1'.

CORL

    Parameters
    ----------
    batch_size: int, default: 64
        training batch size
    input_dim: int, default: 64
        dimension of input data
    embed_dim: int, default: 256
        dimension of embedding layer output
    normalize: bool, default: False
        whether normalization for input data
    encoder_name: str, default: 'transformer'
        Encoder name, must be one of ['transformer', 'lstm', 'mlp']
    encoder_heads: int, default: 8
        number of multi-head of `transformer` Encoder.
    encoder_blocks: int, default: 3
        blocks number of Encoder
    encoder_dropout_rate: float, default: 0.1
        dropout rate for encoder
    decoder_name: str, default: 'lstm'
        Decoder name, must be one of ['lstm', 'mlp']
    reward_mode: str, default: 'episodic'
        reward mode, 'episodic' or 'dense',
        'episodic' denotes ``episodic-reward``, 'dense' denotes ``dense-reward``.
    reward_score_type: str, default: 'BIC'
        type of score function
    reward_regression_type: str, default: 'LR'
        type of regression function, must be one of ['LR', 'QR']
    reward_gpr_alpha: float, default: 1.0
        alpha of GPR
    iteration: int, default: 5000
        training times
    actor_lr: float, default: 1e-4
        learning rate of Actor network, includes ``encoder`` and ``decoder``.
    critic_lr: float, default: 1e-3
        learning rate of Critic network
    alpha: float, default: 0.99
        alpha for score function, includes ``dense_actor_loss`` and
        ``dense_critic_loss``.
    init_baseline: float, default: -1.0
        initilization baseline for score function, includes ``dense_actor_loss``
        and ``dense_critic_loss``.
    random_seed: int, default: 0
        random seed for all random process
    device_type: str, default: cpu
        ``cpu`` or ``gpu``
    device_ids: int or str, default None
        CUDA devices, it's effective when ``use_gpu`` is True.
        For single-device modules, ``device_ids`` can be int or str, e.g. 0 or '0',
        For multi-device modules, ``device_ids`` must be str, format like '0, 1'.


MCSL

    Parameters
    ----------
    model_type: str, default: 'nn'
        `nn` denotes neural network, `qr` denotes quatratic regression.
    num_hidden_layers: int, default: 4
        Number of hidden layer in neural network when `model_type` is 'nn'.
    hidden_dim: int, default: 16
        Number of hidden dimension in hidden layer, when `model_type` is 'nn'.
    graph_thresh: float, default: 0.5
        Threshold used to determine whether has edge in graph, element greater
        than the `graph_thresh` means has a directed edge, otherwise has not.
    l1_graph_penalty: float, default: 2e-3
        Penalty weight for L1 normalization
    learning_rate: float, default: 3e-2
        learning rate for opitimizer
    max_iter: int, default: 25
        Number of iterations for optimization problem
    iter_step: int, default: 1000
        Number of steps for each iteration
    init_iter: int, default: 2
        Initial iteration to disallow early stopping
    h_tol: float, default: 1e-10
        Tolerance of optimization problem
    init_rho: float, default: 1e-5
        Initial value for penalty parameter.
    rho_thresh: float, default: 1e14
        Threshold for penalty parameter.
    h_thresh: float, default: 0.25
        Threshold for h
    rho_multiply: float, default: 10.0
        Multiplication to amplify rho each time
    temperature: float, default: 0.2
        Temperature for gumbel sigmoid
    device_type: str, default: 'cpu'
        'cpu' or 'gpu'
    device_ids: int or str, default '0'
        CUDA devices, it's effective when ``use_gpu`` is True.
        For single-device modules, ``device_ids`` can be int or str, e.g. 0 or '0',
        For multi-device modules, ``device_ids`` must be str, format like '0, 1'.
    random_seed: int, default: 1230
        random seed for every random value


DAG_GNN

    Parameters
    ----------
    encoder_type: str, default: 'mlp'
        choose an encoder, 'mlp' or 'sem'.
    decoder_type: str, detault: 'mlp'
        choose a decoder, 'mlp' or 'sem'.
    encoder_hidden: int, default: 64
        MLP encoder hidden layer dimension, just one hidden layer.
    latent_dim: int, default equal to input dimension
        encoder output dimension
    decoder_hidden: int, default: 64
        MLP decoder hidden layer dimension, just one hidden layer.
    encoder_dropout: float, default: 0.0
        Dropout rate (1 - keep probability).
    decoder_dropout: float, default: 0.0
        Dropout rate (1 - keep probability).
    epochs: int, default: 300
        train epochs
    k_max_iter: int, default: 1e2
        the max iteration number for searching lambda and c.
    batch_size: int, default: 100
        Sample size of each training batch
    lr: float, default: 3e-3
        learning rate
    lr_decay: int, default: 200
        Period of learning rate decay.
    gamma: float, default: 1.0
        Multiplicative factor of learning rate decay.
    lambda_a: float, default: 0.0
        coefficient for DAG constraint h(A).
    c_a: float, default: 1.0
        coefficient for absolute value h(A).
    c_a_thresh: float, default: 1e20
        control loop by c_a
    eta: int, default: 10
        use for update c_a, greater equal than 1.
    multiply_h: float, default: 0.25
        use for judge whether update c_a.
    tau_a: float, default: 0.0
        coefficient for L-1 norm of A.
    h_tolerance: float, default: 1e-8
        the tolerance of error of h(A) to zero.
    use_a_connect_loss: bool, default: False
        flag to use A connect loss
    use_a_positiver_loss: bool, default: False
        flag to enforce A must have positive values
    graph_threshold: float, default: 0.3
        threshold for learned adjacency matrix binarization.
        greater equal to graph_threshold denotes has causal relationship.
    optimizer: str, default: 'Adam'
        choose optimizer, 'Adam' or 'SGD'
    seed: int, default: 42
        random seed
    device_type: str, default: cpu
        ``cpu`` or ``gpu``
    device_ids: int or str, default None
        CUDA devices, it's effective when ``use_gpu`` is True.
        For single-device modules, ``device_ids`` can be int or str, e.g. 0 or '0',
        For multi-device modules, ``device_ids`` must be str, format like '0, 1'.


GraNDAG

    Parameters
    ----------
    input_dim: int
        number of input layer (number of varibles), must be int
    hidden_num: int, default 2
        number of hidden layers
    hidden_dim: int, default 10
        number of dimension per hidden layer
    batch_size: int, default 64
        batch size of per training of NN
    lr: float, default 0.001
        learning rate
    iterations: int, default 10000
        times of iteration
    model_name: str, default 'NonLinGaussANM'
        model name, 'NonLinGauss' or 'NonLinGaussANM'
    nonlinear: str, default 'leaky-relu'
        name of Nonlinear activation function, 'sigmoid' or 'leaky-relu'
    optimizer: str, default 'rmsprop'
        Method of optimize, 'rmsprop' or 'sgd'
    h_threshold: float, default 1e-7
        constrained threshold, if constrained value less than equal h_threshold
        means augmented lagrangian has converged, model will stop trainning
    device_type: str, default 'cpu'
        The target device to run, support 'ascend', 'gpu', and 'cpu'
    device_ids: int, default 0
        ID of the target device,
        the value must be in [0, device_num_per_host-1],
        while device_num_per_host should be no more than 4096
    use_pns: bool, default False
        whether use pns before training, if nodes > 50, use it.
    pns_thresh: float, default 0.75
        threshold for feature importance score in pns
    num_neighbors: int, default None
        number of potential parents for each variables
    normalize: bool, default False
        whether normalize data
    random_seed: int, default 42
        random seed
    norm_prod: str, default 'paths'
        use norm of product of paths, 'none' or 'paths'
        'paths': use norm, 'none': with no norm
    square_prod: bool, default False
        use squared product of paths
    jac_thresh: bool, default True
        get the average Jacobian with the trained model
    lambda_init: float, default 0.0
        initialization of Lagrangian coefficient in the optimization of
        augmented Lagrangian
    mu_init: float, default 0.001
        initialization of penalty coefficient in the optimization of
        augmented Lagrangian
    omega_lambda: float, default 0.0001
        tolerance on the delta lambda, to find saddle points
    omega_mu: float, default 0.9
        check whether the constraint decreases sufficiently if it decreases
        at least (1-omega_mu) * h_prev
    stop_crit_win: int, default 100
        number of iterations for updating values
    edge_clamp_range: float, default 0.0001
        threshold for keeping the edge (if during training)
