DAG: Directed Acyclic Graph

    pa(i) PA(i)   parents of i
    an(i) AN(i)   ancestors of i
    ch(i) CH(i)   children of i
    de(i) DE(i)   descendants of i


-----------------------------------------------------------------------------

chains          X->Z->Y
fork            X<-Z->Y
collider        X->Z<-Y

    chain & form    path blocked by conditioning on Z
    collider        path blocked if one does not condition on Z
                    path unblocked if one conditions on Z

X and Y are d-separated by a set Z if each path X~~>Y is blocked by Z




Bayes:     P(X,Y) = P(X|Y)P(Y) = P(Y|X)P(X)


X, Y independent  

    X ind Y

    P(X)   = P(X|Y)
    P(Y)   = P(Y|X)
    P(X,Y) = P(X)P(Y)

X and Y are conditionally independent given Z

    X ind Y | Z

    P(X,Y|Z) = P(X|Z)P(Y|Z)

independence in the distribution|graph

    X ind_P Y
    X ind_G Y

    - two nodes are unconditionally (or marginally) independent in the graph when there’s no open path that connects them directly or indirectly.
    - two nodes, X and Y, are conditionally independent given (a set of) node(s) Z when Z blocks all open paths that connect X and Y.

causal Markov condition (also known as the causal Markov assumption or (local) Markov property).

    Vi is independent of all its non-descendants (excluding its parents) given its parents
    
    Vi ind Vj | PA(Vi) for All Vj in V - DE(Vi) - PA(Vi)