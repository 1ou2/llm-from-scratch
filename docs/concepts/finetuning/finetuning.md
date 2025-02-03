On veut changer les paramètres du modèles. 
- augmentation et adaptation à un 
# Supervised fine tuning
Nouveaux jeu de données, modifications des poids.
1. on prend un foundation model
2. finetuning avec des données spécifiques -> cela va changer un nombre réduit de poids dans le modèle, ou ajout d’un nombre réduit de paramètres

# Distillation
Utilisation d’un large modèle, on le combine avec un dataset spécifique.
On génère un student model.
Il a la qualité du modèle parent sur les données spécifiques
Modèle beaucoup plus petit : moins de latence, moins de cout et spécialisé sur use case.
Pour le processus de distillation
- modèle teacher : on lui donne des données du dataset. Il génère une réponse. On ne prend pas juste la réponse mais toutes la distribution de probabilités.
- le modèle student : on lui donne les données du dataset en entrée, et il doit chercher à optimiser toute la distribution de probabilité. 

C'est donc beaucoup plus efficace qu'un entrainement classique. On optimise plus de paramètres à la fois, et on transmet plus de savoir entre le teacher et le student

# Reinforcement learning
Le principe est d'avoir un état, des actions et une fonction de récompense qu'on cherche à optimiser.
Dans le cadre des LLM, habituellement on a :
- phase d'entrainement non-supervisé/autosupervisé. On a un énorme dataset de phrases, on donne des bouts de texte et le modèle doit trouver le token qui suit.
- phase fine-tuning. On a un dataset de questions et de réponses. On aligne le modèle sur ce qu'on veut faire. On l'optimise à donner des réponses quand on lui pose une question. À être poli, etc...
- reinforcement learning with human feedback : On a une fonction de récompense (reward) dont le score est attribué par un humain. On demande au modèle de générer une réponse à une question et c'est l'humain qui donne la note. Mais comme c'est difficile d'évaluer une réponse de façon objective on demande à l'humain de classer les réponses.
Si on demande :
Q: Combien fait 2+2?
R1: 4
R2: La réponse est 4
R3: c'est facile.
Est-ce que je donne une bonne note à R1, ou est-ce que je veux que la réponse soit plus longue comme R2. Donner une note dans l'absolu est infaisable avec des humains.
On demande donc à l'annotateur humain de classer les réponses par ordre de préférence.
Q: Où est shangai ?
R1 : Shangai est en chine.
R2 : shangai est une jolie ville
R3 : j'aime les pizzas
R1 > R2 > R3

Pour avoir cette fonction de récompense, on part donc du jeu de données annotés par des humains :
- jeu de données avec des questions
- on demande au modèle de générer plusieurs réponses en utilisant une températuer élevée
- un humain, classe les réponses par ordre de préférence.

Puis on part du base model, on le modifie sur le dernier layer. On ne veut plus retourner un softmax qui donne une distribution de probablité pour chaque token, à la place on remplace par un layer qui retourne une seule valeur représentant notre reward.

L'objectif de l'entrainement est de maximiser cette reward.
La fonction de reward est -log(sigmoid(reward(good_response) - reward(mauvaise_réponse)))
Au lieu d'une fonction de perte, on parle de fonction d'objectif car c'est une valeur qu'on veut maximiser et non minimiser.
Si le modèle est performant on aura
 - Si R = rereward(good_response) - reward(mauvaise_réponse) positif, donc sigmoid(R) sera entre 0 et 1, et alors - log (sigmoid(R)) sera positif
 - Si R = rereward(good_response) - reward(mauvaise_réponse) négatif, donc sigmoid(R) sera entre -1 et 0, et - log (sigmoid(R)) sera négatif

## Deepseek
Pour deepseek R1, le principe utilisé est 
- on part d'un base modèle deepseek v3 déjà fine-tuné
- puis on fait une phase de reinforcement learning automatique
    - état : distribution de probabilité des tokens du LLM
    - reward : il y a deux types de reward, est-ce que la syntaxe de la réponse contient des balises <think> et est-ce que la réponse est correcte. Dans leur cas, ils donnent au modèle des problèmes de mathématique (on connait la réponse), et des problèmes de codage (type leetcode), et alors le code produit est exécuté et on exécute des tests unitaires pour vérifier que le code est correct.

# Quantization


# Lora