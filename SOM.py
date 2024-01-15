import numpy as np
import sompy

# Defina o conjunto de dados
data = np.array([
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
    [1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Configurar e treinar a Rede SOM
map_size = (5, 5)  # Tamanho da grade SOM
som = SOMFactory.build(data, mapsize=map_size, initialization='random')
som.train(n_job=1, verbose='info')  # n_job=1 para evitar problemas com alguns ambientes

# Obter a visualização da grade SOM
from sompy.visualization.mapview import View2D
view2D = View2D(map_size[0], map_size[1], 'SOM Map')
view2D.show(som, col_sz=4, which_dim='all', desnormalize=True)