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

## cross entropy
Cette fonction attend les channels dans la deuxième dimension
Il veut une matrice B,C,T
Cela calcule la log likelyhood. Or dans un modèle de bigram à 65 caractères, avec une initialisation au hasard on s’attend à une perte égale à ln(1/65) soit 4.17.