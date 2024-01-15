import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt

# Seu conjunto de dados aqui
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

data = np.transpose(data)


# Rótulos dos animais
animal_labels = ['Dove', 'Hen', 'Duck', 'Goose', 'Owl', 'Hawk', 'Eagle', 'Fox', 'Dog', 'Wolf', 'Cat', 'Tiger', 'Lion', 'Horse', 'Zebra', 'Cow']

# Configurar e treinar a Rede SOM
map_size = (16, 13)
som = MiniSom(map_size[0], map_size[1], data.shape[1], sigma=0.4, learning_rate=0.5)
som.train_random(data, 10000)

# Visualização
plt.figure(figsize=(8, 8))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # Mapa de distâncias entre neurônios
plt.colorbar()

# Ajuste de espaçamento vertical
vertical_spacing = 0.2
print(som.distance_map().T)

for i, x in enumerate(data):
    w = som.winner(x)
    plt.text(w[0] + 0.5, w[1] + 0.5 + i * vertical_spacing, animal_labels[i], color='red', fontsize=8, ha='center', va='center', rotation=45)  # Ajuste de rotação para melhorar a legibilidade

plt.title('Mapa Auto-Organizável (SOM)')
plt.show()
