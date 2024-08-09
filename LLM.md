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
Soit P un tensor 27/27
P.sum(0,keepdim=True).shape est un Tensor 1x27 -> c’est le résultat de la somme des colonnes. C’est donc comme si on rajoutait une ligne de somme après les colonnes.
P.sum(1,keepdim=True).shape est un Tensor 27x1 -> c’est le résultat de la somme des lignes. C’est donc comme si on rajoutait une colonne de somme après la dernière ligne.
Avec keepdim=False, on n’a plus un Tensor avec 2 dimensions, mais un Tensor de taille 27.
C’est la même diff, en taille 3 que
- [[1],[2],[3]] -> dimension = 3x1
- [1,2,3]       -> dimension = 3

## Randn
Génère des nombres aléatoires en utilisant une distribution normalisée (courbe en cloche)

# Bigram
C'est un modèle prédictif simple où on calcule à partir d'un jeu de données, quelles sont les probabilités associés à chaque bigram, c'est à dire à chaque couple de deux lettres consécutives.
On définit le token "." comme étant un token spécial marquant le début et la fin d’un mot.
## Probabilité
Pour chaque mot, décomposer et caculer le nombre d'occurences du bigram. 
Soit N, un Tensor 27x27 represantant toutes les occurences des bigrams dans notre jeu de données
N[0] : represente les occurences des deux premières lettres .a, .b, .c etc...
On calcule la probabilité, étant donné une lettre d’avoir la suivante.
Soit P un Tensor 27x27 representant cette probabilité
Pour calculer P, on parcourt N ligne par ligne et on normalise les lignes.
P[i] = N[i] / sum(N[i])
Attention avec cette méthode on peut avoir des 0. Par exemple on n’a pas jq dans le jeu de données. Donc dans N, le compte est à 0 est la probabilité est à zéro.
Cela va poser problème quand on va calculer la fonction de perte (loss function), car on prend le log, et log(0) = - infinity.
Pour résoudre ce problème on utilise une smoothing function.
On rajoute articificiellement une valeur au comptage.
On peut par exemple ajouter 1 à toutes les celles de N. 

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
C’est en minimisant NLL qu’on évalue la qualité du modèle.

# Bigram Language Model
Réseau de neurone, qui à partir d’un caractère prédit le suivant.
On va changer nos poids, via l’algo de la descente de gradient de façon à minimiser la loss function.

## One hot encoding
Quand on doit manipuler des integer, cela ne va pas trop faire de sens, dans nos calculs de descente de gradient où on fait des opérations qui s’appliquent plutôt à des float.
une technique consiste à transforme un integer en un tensor.
Si on a les nombres 0 à 5 et qu’on veut encoder le nombre 2, on utilisera le vecteur [0,0,1,0,0] où le chiffre 1 représente l’index du nombre à encoder.
```python
import torch.nn.functional as F
xenc = F.one_hot(tx, num_classes=27).float()
```

## Tactique
On veut mettre en place un réseau de 27 neurones, qui simule l'algorithme statistique calculé sur les bigram.
On voudrait que le réseau donne une probabilité à un caractère donné.
- Si, on veut que les Weights représentent la matrice P des bigram on n'y arrivera pas. Or on ne peut pas avec un réseau de neurone avoir une telle matrice directement. La matrice de stat, sont des nombres entre 0 et 1 tels que la somme des lignes vaut 1.
- idem si on veut arriver à la matrice N de comptage des bigrams. Ce sont des entiers.


Ces poids vont nous donner des logs counts.
On va considérer que W représentent les outputs de la fonction log, et on va appliquer l'opération exponentielle.
0 < exp(x) < 1 pour x < 0
exp(x) > 1 pour x > 0

Puis pour chaque ligne, on normalise.

Cette opération s'appelle **softmax**:
- à partir de nombre réel (float négatifs et positifs)
- on applique exp(x)/(Sum(exp(x)))
- on obtient une distribution de probabilité -> que des nombres entre 0 et 1 dont la somme fait 1

[x] @ [W] + [b]= loss function 


