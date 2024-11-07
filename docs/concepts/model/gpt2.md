# Traitement par blocs
Les données dans un transformer ne sont pas utilisées dans leur totalité. On doit la partitionner en chunks ou blocs. Typiquement on aura en entrée un maximum de ```block_size``` tokens.
Mais durant l’entrainement on va donner au transformer des entrées allant de 1 à ```block_size``` tokens, pour qu’il ait aussi des entrées avec peu de tokens et qu’il génère quand même une suite probable. Lors de l’inférence on veut pouvoir commencer avec un seul token.

Durant l’entrainement, on a :
- la dimension temporelle, avoir en entrée 1 token, puis 2, puis 3 jusqu’à block_size
- la dimension batch : on va regrouper x exemples ensemble pour optimiser les calculs et paralléliser, et maximiser l’utilisation des GPUs. Les chunks sont traités de façon complètement indépendante.

# batch
On a ```batch_size``` : nombre de séquences qu’on traite en // durant la forward et la backward pass

# Torch
## nn.Embeddings(t,d)
Il s’agit d’une légère surcouche pour faire un tensor de dimension (t,d).
Par défaut les valeurs sont initialisées au hasard
```
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # (65,65)

# idx and targets are both (B,T) tensor of integers
# (B,T,C) = (4,8,65)
# B : batch -> nombre d’inputs en parallèle
# T : time -> longueur max du contexte
# C : channels -> dimension de l’embedding
logits = self.token_embedding_table(idx) # (B,T,C) = (4,8,65)
```

explication :
On a l’embedding qui est une matrice 65x65. À chaque ligne (représentant un id de token) est associé le score/prédiction du prochain caractère.
[![](images/embedding.png)](images/embedding.png)
## cross entropy
Cette fonction attend les channels dans la deuxième dimension
Il veut une matrice B,C,T
Cela calcule la log likelyhood. Or dans un modèle de bigram à 65 caractères, avec une initialisation au hasard on s’attend à une perte égale à ln(1/65) soit 4.17.

# Bigram  model
On a juste un Embeddings qu’on entraine.

# Transformer
On a nos tokens, et on a layer linéaire pour créer un embedding de ces tokens
## Token embedding
self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
## Postion embedding
self.position_embedding_table = nn.Embedding(block_size, n_embed)
## neurones de sorties
self.lm_head = nn.Linear(n_embed, vocab_size)

# Tête d’attention - mécanisme de sef attention
On a un paramètre qui est la ```head_size = 16```. C’est la taille que l’on donne à notre matrice d’attention.
Dans notre transformer, on a en entrée un matrice x de taille (B,T,C)
- B : taille du batch, combien d’éléments sont traités en parallèle
- T : nombre de tokens du block
- C : nombre de channels par token, correspond à la dimension de l’embedding
Création de deux couches de neurones :
key = nn.Linear(C,head_size, bias=False) # (C,16)
query = nn.Linear(C,head_size, bias=False) # (C,16)

Pour chaque batch, on calcule deux vecteurs
- k = key(x)
- q = query(y)
On a alors pour k, et q des tenseurs de taille (B, T, 16). On a B tenseurs dans notre batch. Et chaque tenseur a pour taille (T,16).
On commence par calculer leur produit scalaire. Cela donne la proximité entre k et q.
wei = q. transpose(k)
Puis on fait un masque sur tous les éléments du tensor qui sont dans la diagonale supérieure.
On ne veut pas calculer la relation entre un token, et token d’un indice plus grand, car les tokens ont un ordre et un token de plus grand indice arrive plus tard dans la phrase. Si on calcule son impact/sa relation sur un mot cela revient à dire qu’on connait le mot suivant et donc il est facile de le prédire !
Techniquement on positionne à -infini la valeur car on va appliquer un softmax qui va donc passer cette valeur à 0.
wei = 
La matrice d’attention A se calcule alors par :
- A = softmax(q . transpose(k))
On fait donc (B, T, 16) @ (B , 16, T) -> (B, T, T)
Enfin, une fois qu’on connait le poids relatif des tokens les uns par rapport aux autres on peut multiplier par la valeur du token
out = wei @ x # (B, T, T) @ (B, T, C) -> (B,T,C) = (4,8,32)

Interpretation
x est l’embedding du mot + sa position.
Une fois passé par le mécanisme d’attention, on a un nouvel embedding qui represente le mot + un delta issuu de sa relation avec les autres mots.
Exemple: si on le mot ```avocat```, et que dans la phrase il y a le mot ```assiette```, on peut imaginer que l’embedding du mot va être modifié dans la direction du sens nourriture.
Alors que si on a ce mot avec le mot ```tribunal``` a côté, alors l’embedding vectoriel sera modifié dans le sens de la justice et du droit.

## code
```
    torch.manual_seed(1337)
    B,T,C = 4,8,32 # batch, time, channels
    x = torch.randn(B,T,C)
    head_size = 16
    key = nn.Linear(C, head_size, bias=False)
    query = nn.Linear(C, head_size, bias=False)
    value = nn.Linear(C, head_size, bias=False)
    k = key(x)   # (B, T, 16)
    q = query(x) # (B, T, 16)
    v = value(x) # (B, T, 16)
    wei = q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) = (B, T, T) = (4, 8, 8)

    tril = torch.tril(torch.ones(T, T))
    wei = wei.masked_fill(tril == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1) # (B, T, T) = (4, 8, 8)
    out = wei @ v # (B, T, T) @ (B, T, 16) -> (B,T,16) = (4,8,16)
```

## Matrice d’attention
```
       [[ 0.4516,  0.3215, -3.1926,  0.3077, -0.6161,  0.2563, -0.2989, -2.1917],
        [-0.4001, -0.9621,  1.9568,  0.6661, -0.3263,  0.2626, -1.3973, -0.8945],
        [-0.4620,  0.5860, -4.6738, -0.3218,  1.2684, -0.1740,  1.2461, -2.2283],
        [-0.7175, -1.0279, -2.0509, -2.7234,  0.3123, -0.1642,  1.5162, -0.7767],
        [-0.4039,  0.5160, -2.0697, -0.4098, -0.8053,  0.5221, -0.4124,  1.3377],
        [ 0.8232,  3.0237, -3.0655,  0.7040,  0.6721, -0.4669,  2.3746,  0.3118],
        [-1.4141, -1.4241, -0.8039, -1.7450, -0.7403,  0.9819, -0.9006, -2.3158],
        [-0.5028,  1.6844, -0.4185,  1.0239,  1.0275,  0.1398,  0.4882,  1.5573]]


       [[ 0.4516,    -inf,    -inf,    -inf,    -inf,    -inf,    -inf,    -inf],
        [-0.4001, -0.9621,    -inf,    -inf,    -inf,    -inf,    -inf,    -inf],
        [-0.4620,  0.5860, -4.6738,    -inf,    -inf,    -inf,    -inf,    -inf],
        [-0.7175, -1.0279, -2.0509, -2.7234,    -inf,    -inf,    -inf,    -inf],
        [-0.4039,  0.5160, -2.0697, -0.4098, -0.8053,    -inf,    -inf,    -inf],
        [ 0.8232,  3.0237, -3.0655,  0.7040,  0.6721, -0.4669,    -inf,    -inf],
        [-1.4141, -1.4241, -0.8039, -1.7450, -0.7403,  0.9819, -0.9006,    -inf],
        [-0.5028,  1.6844, -0.4185,  1.0239,  1.0275,  0.1398,  0.4882,  1.5573]]




       [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.6369, 0.3631, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2586, 0.7376, 0.0038, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.4692, 0.3440, 0.1237, 0.0631, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.1865, 0.4680, 0.0353, 0.1854, 0.1248, 0.0000, 0.0000, 0.0000],
        [0.0828, 0.7479, 0.0017, 0.0735, 0.0712, 0.0228, 0.0000, 0.0000],
        [0.0522, 0.0517, 0.0961, 0.0375, 0.1024, 0.5730, 0.0872, 0.0000],
        [0.0306, 0.2728, 0.0333, 0.1409, 0.1414, 0.0582, 0.0825, 0.2402]]
```



[[0.0972, 0.0573,     -0.1047,     -0.0467,     -0.1401,
             -0.8413,     -0.1362,     -0.6747,     -0.2154,      1.0993,
              0.2343,      0.0326,     -0.1852,      0.1478,     -0.6104,
              1.5391],
        [     0.3612,     -0.6797,     -0.7709,      0.6483,     -0.2445,
             -0.5790,     -1.5354,     -0.7219,     -0.1883,      0.0109,
              0.2399,     -0.0545,     -0.1437,      0.0493,     -0.8864,
              0.7240],
        [    -0.1098,      0.8060,      0.8114,     -0.3400,     -0.4584,
              0.0054,      1.3075,     -0.7778,     -0.6282,      0.0742,
             -0.2187,      0.1813,     -0.2085,      0.6720,      0.0694,
              0.9866],
        [     0.3043,      1.1563,      0.1380,     -2.0818,     -0.1047,
              0.5229,      1.2301,      0.5365,     -0.9001,     -1.0794,
             -0.2433,      0.0010,      0.2483,      0.0442,     -0.6785,
             -0.3334],
        [    -0.5300,     -0.9214,      0.3791,     -0.0207,      0.3733,
             -0.1613,     -0.7093,      0.0420,      0.1615,      0.1662,
              0.5669,      0.5506,     -0.0711,     -0.5554,     -0.1208,
             -0.4528],
        [    -0.6965,      0.4446,      0.8095,     -0.6036,      0.0479,
             -0.4640,     -0.2097,      0.5598,      0.5720,      0.3643,
              0.0594,     -1.3565,      0.6867,      0.5451,     -0.6737,
              0.6352],
        [     0.3546,      0.1157,     -0.4229,     -0.4704,     -0.2267,
              0.1567,     -0.2100,     -1.0505,     -1.0665,     -0.8319,
              0.1989,      0.9078,      0.3519,      0.0566,     -0.6488,
              0.0551],
        [    -1.7223,      0.5108,      0.2968,      0.2329,      0.2418,
              0.3372,     -0.2523,      0.6476,     -1.4068,     -0.6438,
              0.0745,     -0.5873,      0.1296,     -0.2159,     -0.7506,
              0.3231]]