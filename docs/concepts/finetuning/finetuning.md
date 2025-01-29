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
- reinforcement learning with human feedback : On a une fonction de récompense (reward) dont le score est attribué par un humain. On demande au modèle de générer une réponse à une question et c'est l'humain qui donne la note.

Pour deepseek R1, le principe utilisé est 
- on part d'un base modèle deepseek v3 déjà fine-tuné
- puis on fait une phase de reinforcement learning automatique
    - état : distribution de probabilité des tokens du LLM
    - reward : il y a deux types de reward, est-ce que la syntaxe de la réponse contient des balises <think> et est-ce que la réponse est correcte. Dans leur cas, ils donnent au modèle des problèmes de mathématique (on connait la réponse), et des problèmes de codage (type leetcode), et alors le code produit est exécuté et on exécute des tests unitaires pour vérifier que le code est correct.

# Quantization


# Lora