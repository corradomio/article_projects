Graph
    order   n of vertices
    size    n of edges


Generazione del dataset
-----------------------

    1) generazione del grafo diretto aciclico
        n. nodi              10-25/5
        n. archi            .05-.1
        tipo di generatore
    2) generazione del dataset, basandosi sulla libreria gcastle


IIDSimulation
-------------

    W: np.ndarray
        Weighted adjacency matrix for the target causal graph.
    n: int
        Number of samples for standard trainning dataset.
    linear
        sem_type: gauss, exp, gumbel, uniform, logistic
    nonlinear
        sem_type: mlp, mim, gp, gp-add, quadratic
    noise_scale
        Scale parameter of noise distribution in linear SEM



Grafi
-----
    Per gli ordini 2,3,4,5, TUTTI i DAG
    per gli ordini 10, 15, 20, 25
        1000 DAG con densita':  10%, 15%, 20%
        (quindi 3000 DAG)


HDF5
----

    /<deg>/<graph_id>
        attrs:
            'n':                n of nodes (order)
            'm':                n of edges (size)
            'adjacency_matrix': adjacency matrix (n x n)
            'wl_hash':          Weisfeiler Leman hash
        <method>/<sem_type>:    dataset (10000 x n)


    <deg>:      '2', '3', '4', '5', '10', '15', '20', '25'
    <method>/<sem_type>:
        "linear": ["gauss", "exp", "gumbel", "uniform", "logistic"],
        "nonlinear": ["mlp", "mim", "quadratic"]  # "gp", "gp-add",

        Note: "gp", "gp-add" removed because the data generation is too time expensive