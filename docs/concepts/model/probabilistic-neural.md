# A Neural Probabilistic Language Model
https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

L’objectif de ce papier est de présenter un modèle de langage probabiliste basé sur les réseaux de neurones. Il s’agit d’associer une probabilité au mot suivant à prédire.

Ce papier décrit un modèle de réseau de neurone qui apprend simultanément:
- une représentation des mots dans un espace vectoriel ( embedding de mot ) à partir d’un corpus de textes
- la fonction de probabilité d’un mot suivant à partir de la représentation des mots dans l’espace vectoriel

# Architecture
On a une séquence de mots en entrée, et à partir de cette séquence on essaie de prédire le prochain mot. 

## preparation des données
Avant d’y arriver, on transforme les mots en token, et techniquement ce que le modèle va donner c’est la probabilité du prochain token.
Texte d’entrée -> découpage en mots, chiffres, symboles -> transformation en liste de tokens (int).
Exemple, à partir d’un extrait de notre corpus d’entrainement
"heures venaient de tinter" -> "heures ven aient de t inter" -> 743, 345, 284, 181, 85, 855
Le modèle prend en entrée une séquence de tokens de longueur fixe. Si on fixe notre entrée à 5, cela veut dire qu’à partir des 5 premiers tokens on veut prédire le suivant (le token 6)
Notre input X est issu de "heures ven aient de t" X = [743, 345, 284, 181, 85] et on veut prédire Y = [855] soit "inter"
On parle de ```block_size``` ou de taille de contexte. Ici block_size = 5
```X = [743, 345, 284, 181, 85] -> Y = [855]```

Pour paralléliser les calculs, on va créer des batchs regroupant plusieurs données d’entrées. Par exemple, si on prend un batch_size = 32, alors on créé 32 vecteurs X, et 32 vecteur Y.
x_batch.size = [32,5] -> on a 32 lignes de 5 entrées. Chaque ligne contient 5 tokens
y_batch.size = [32] -> 32 lignes

## Feature word vector
L’étape suivant est la création d’une matrice d’embedding des tokens. Chaque token va être représenté par un vecteur de dimension m.
La taille de vocabulaire ```vocab_size```, correspond au nombre total de tokens différents qui existent. Si on a ```vocab_size tokens```, et que chaque token est représenté par un vecteur de taille ```m```, alors on a une matrice C de taille ```(vocab_size, m)```
t = vocab_size # nombre de tokens
On va ensuite les transformer en embeddings.
Pour cela on a la matrice C de taille (t, d) où m est la dimension de l'espace vectoriel.
C donne pour chaque tokens sa représentation vectorielle. Par exemple si on a un vocab_size = 1000 et m = 30, cela signifie qu’on va représenter chaque tokens par un vecteur de dimension 30.
Chaque ligne de C donne la représentation vectorielle d’un token.
On aura 1000 lignes et 30 colonnes.
Trouver l’embedding du token 17 revient à prendre la ligne 17 de C.
C[17] = [d1,d2, ..., d30]

On n’utilise pas la matrice C, unitairement. On passe en paramètres p tokens. 
Dans nos données d’entrainements on construits des vecteurs de taille p.
Chaque vecteur est composé des tokens.
Si on veut trouver le mot suivant en connaissant les 5 mots précédents, on aura p=5, c’est la taille du contexte et 
"heures venaient de tinter" -> "heures ven aient de t inter" -> 743, 345, 284, 181, 85, 855
On veut à partir des 5 premiers tokens  prédire le suivant
Notre input X est issu de "heures ven aient de t" X = [743, 345, 284, 181, 85] et on veut prédire Y = [855] soit "inter"
Pour optimiser on va prendre plusieurs phrases d’entrée en parallèle et construire un vecteur X contenant 


