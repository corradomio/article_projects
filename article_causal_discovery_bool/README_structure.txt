[container]
    /[order:str]/[wl_hash:str]
        adjacency_matrix: [order:int] x [order:int]
        n: int          (order)
        m: int          (size)
        wl_hash:str
        /dataset: Dataset[100 x 3000 x [order:int]]


finfos: {
    [order:str]: {
        [wl_hash:str]: [
            {
                'instance': int
                'n': int
                'm': int
                'wl_hash': str
                'nodes': {
                    [id_node: int]: {
                        'f': [...],
                        'n': [id_node:int]
                        'noisep': [prob:float]
                        'params': [id_node, ...]
                    }
                }
            },
            ...
        ],
    ...
}
