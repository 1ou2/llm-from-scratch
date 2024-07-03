Un réseau de neurone est expression mathématique :
- On a des données d’entrée (inputs),
- une matrice dont on essaie de calculer les valeurs (weights) durant la phase d’entrainement
- et on calcule un résultat : soit une prédiction soit une loss function

Pour  calculer les weigths on essaie de minimiser une fonction "loss function", car plus on minimise, et moins on a d’écart entre le résultat du réseau et le résultat attendu.
L’algorithme qui permet itérativement de converger vers ce minimum s’appelle "backpropagation". C’est un algorithme efficace pour évaluer le grandient.

## Forward pass
Étant donné des poids (weights) et des donnés d’entrées (inputs), le cacul de la sortie (output) se fait en faisant une forward pass.
On part des données d’entrées, et on calcule à chaque étape du réseau de neurones les valeurs intermédiaires jusqu’à arriver au résultat final.

## Backpropagation vs Gradient descent
- Gradient descent : algorithme d’optimisation général pour calculer les poids du modèle.
- Backpropagation : c’est une étape de l’algorithme gradient descent, où on met à jour les poids du modèle en calculant des derivées partielles (le gradient) qui donne l’ajustement qu’on donne au poids pour converger vers un minimun.
  
À chaque itération, on a un ajustement des poids. 
- Soit [W]^n les poids à l’étape n
- Lr : learning rate - un paramètre qui dit à quelle vitesse on veut aller (si c’est trop petit, on converge tout doucement, si c’est trop grand on peut osciller et ne pas trouver le min)
- [G]^n : le gradient - c’est à dire le petit ajusteemnt local des poids qui va permettre au modèle de se rapprocher de l’optimal
 [W]^n+1 = [W]^n - Lr * [G]^n

 

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
Dans Pytorch, un Tensor est un objet qui a des méthodes, une représentation interne et qui a donc un ensemble d’opérations disponibles de façons optimisése.

# Questions
- différence entre un vecteur et un tensor : pourquoi deux noms différents
- dans un réseau de neurone, on a les poids qui représentent les valeurs des opérations à effectuer mais où sont stockés les opérations. Ex je veux passer du layer 3 à 4. Et je dois faire 3x-7. Où est stocké cette équation ?
- à quoi sert la "forward pass" vs la "backward pass" dans le gradient descent
