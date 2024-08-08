# Torch
## Generator
Utilisation d'un generator permet d'avoir des résultats reproductibles même en cas d'utilisation de la fonction random.

g = torch.Generator().manual_seed(213) 
p = torch.rand(3,generator=g)

On est assuré de toujours avoir les 3 mêmes nombres pour p
## Multinomial
Si on veut un échantillon d'une distribution de probabilité, on utilise la fonction torch.multinomial
- tu me donnes des probabilités
- et je te donne un jeu de données qui va respecter cette distribution

## Broadcasting 
Attention, il faut se poser la question si une opération est possible. Par exemple est-ce que je peux diviser un Tensor P de dimension 27x27 par un Tensor de dimension 1x27.
C'est ce qui s'appelle broadcasting.


# Bigram
C'est un modèle prédictif simple où on calcule à partir d'un jeu de données, quelles sont les probabilités associés à chaque bigram, c'est à dire à chaque couple de deux lettres consécutives.



## Probabilité
Pour chaque mot, décomposer et caculer le nombre d'occurences du bigram

## Calcul via des Tensors
Attention, il faut se poser la question si une opération est possible. Par exemple est-ce que je peux diviser un Tensor P de dimension 27x27 par un Tensor de dimension 1x27.

## Loss function
Pour calculer la qualité de notre modèle il faut multiplier toutes les probabilités de chaque bigram entre elles.
Ce nombre va tendre vers 0, car on mulitplie des probas p telles que 0 < p < 1
loss = P[b1]*P[b2]*...P[bn] 
### Log
Il est plus simple de calculer, le log de ce nombre, qui sera une somme.
log(a*b*c) = log a + log b + log c
### Negative log likelyhood
À partir du log on peut le moyenner par le nombre d'éléments (le nombre de bigram)
Ensuite, le log sera négatif car log tend vers -infini en 0, et vers 0 en +infini
On utilise donc souvent le negative log likelyhood
nll = - (logP[b1] +P[b2]+...P[bn] )/n
L'avantage du nll est que c'est un nombre positif, et plus il tend vers 0 plus la prédiction de notre modèle est exacte.

