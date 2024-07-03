Un réseau de neurone est expression mathématique :
- On a des données d’entrée (inputs),
- une matrice dont on essaie de calculer les valeurs (weights) durant la phase d’entrainement
- et on calcule un résultat : soit une prédiction soit une loss function

Pour  calculer les weigths on essaie de minimiser une fonction "loss function", car plus on minimise, et moins on a d’écart entre le résultat du réseau et le résultat attendu.
L’algorithme qui permet itérativement de converger vers ce minimum s’appelle "backpropagation". C’est un algorithme efficace pour évaluer le grandient.

# Micrograd
Bibliothèque python permenttant d’illustrer et de calculer des gradients
- Encapsulation des valeurs dans des objets Value
- opération +,-,*,/ et puissance (**)
- permet de calculer la backpropagation
- fonction backward() : initie le calcul de backpropagation, on va calculer toutes les dérivées des expressions, qui ont amenées à cette valeur. Ex: si g dépend de a,b,c,d,e,f, avec a et b valeurs d’entrées et c,d,e, et f des valeurs intermédiaires on va pouvoir calculer la dérivée de g par rapport à a et donc savoir comment on doit faire évoluer a pour que g varie dans le "bon sens". Si on veut diminuer g, on saura comment modifier a et b pour diminuer g

Exemple :
Si dg/da = 130 alors une petite augmentation de a correspond à une augmentation avec une pente de 130

# Dérivée
Dans un réseau de neurone on ne cacule pas symboliquement le résulat des dérivés à l’aide d’équations. On a des dizaines de milliers de paramètres, on ne pose pas ces équations.
Mais on peut aussi voir la dérivée, comme la pente locale de la fonction. Cela se calcule numériquement 
- soit la fonction f(x)
- soit h un incrément tendant vers 0
- l’approximation de la dérivée est : (f(x+h) - f(x))/h quand h tend vers 0

## Scalar vs Tensor
Micrograd permet de travailler sur des scalaires (des floats), en production pour des raisons de performances on parallélise les calculs et les bibliothèques telles que pytorch ne prennent pas en entrée des scalaires mais des Tensors. Un Tensor est un vecteur.


# Questions
- différence entre un vecteur et un tensor : pourquoi deux noms différents
- dans un réseau de neurone, on a les poids qui représentent les valeurs des opérations à effectuer mais où sont stockés les opérations. Ex je veux passer du layer 3 à 4. Et je dois faire 3x-7. Où est stocké cette équation ?
- à quoi sert la "forward pass" vs la "backward pass" dans le gradient descent