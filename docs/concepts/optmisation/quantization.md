# quantization
La question qu’on se pose c’est comment réduire la taille de notre modèle. Les LLM ont des milliards de paramètres. Cela prend de la place sur le disque, cela necessite beaucoup de mémoire, et en inférence cela prend du temps de générer des tokens.
Une technique d’optimisation s’appelle la quantization. On veut « discretiser » les paramètres.
Les paramètres du réseaux de neurones sont stockés sous forme de nombre réels. Le stockage utilisé pour les weights et bias est floating point number.

L’objectif de la quantization est de transformer les nombres à virgules flottantes en entier. On le fait pour deux raisons :
- réduction de la taille du stockage FP32 necessite 4 octets, alors que INT8 n’en necessite qu’un
- accélération des calculs matriciels. L’essentiels des opérantionns sont des matmul (multiplications avec des additions) et c’est calcul sont bien plus rapides sur des entiers que sur des flottants.

## FP32
un nombre flottant est stocké sous la forme 
- 1 bit de signe
- 8 bits d’exposants
- 23 bits de mantisse

value = (-1)^sign * 2^(exposant - 127) * (1 + sum(mantisse_i * 2^-i)

# Post training quantization
La plupart du temps on réalise le processus de quantization après la phase d’entrainement.
On cherche donc une formule qui à un FP32 associe un INT8 par exemple. Mais on peut choisir n’importe quelle intervalle.
Note, les GPU et CPU sont optimisés pour les calculs sur 8,16,32 et 64 bits. On ne va jamais choisir de faire une quantization sur 11 bits !

Processus : on cherche la valeur min et la valeur max. On associe la valeur min réelle à notre plus petit nombre et la valeur max à notre entier max.
On calcule deux constantes :
- S où scale qui est le ratio permettant de projeter l’intervalle initial flotant sur l’intervalle final entier
- Z où le point 0, qui sera l’offset.

Grâce à ces valeurs S et Z on peut à partir des valeurs quant retrouver une valeur approchée des valeurs initiales flottantes.

Pour chaque matrice, chaque layer on va calculer les couples (S,Z), on va faire la quantization.
Note les inputs sont aussi quantizés. Pour calculer l’échelle on réalise une calibration. C’est à dire qu’on fait tourner l’inférence sur un ensemble de jeu de données pour calculer les S,Z de nos entrées.

On peut aussi calculer les S,Z pour des blocs dans nos matrices. On aura une plus grande précision, moins de perte d’approximation mais il faut stocker ces deux valeurs, et puls on a de blocs plus cet overhead va s’ajouter.

Après chaque couche du réseau, on fait l’opération de dequantization avant d’aller sur la couche suivante.

# FAQ - Quantization != arrondi
On ne fait pas un arrondi de la valeur en FP8, on passe d’un intervalle réel à un intervalle avec des valeurs entières.

# FAQ - symetric vs asymetric
Suivant la méthode on peut choisir une quantization symétrique ou assymétrique. Dans la version symétrique la valeur du 0 est préservée. Le FP 0.0 est toujours associé au nombre 0.
Cela a un intérêt pour les réseaux de convolutions ou cette optimisation est importante.
Et on peut toujours passer d’un intervalle non-symétrique à un intervalle en l’agrandissant.
Par exemple, [-3,2] est asymétrique mais si on l’aggrandit à [-3,3] il devient symétrique.
L’inconvénient c’est que le range [2,3] ne sert à rien. On n’aura aucune valeur dans cette partie. Et au final on va perdre en précision sur le reste de l’intervalle.

# Notes
https://github.com/google/gemmlowp/blob/master/doc/quantization.md
