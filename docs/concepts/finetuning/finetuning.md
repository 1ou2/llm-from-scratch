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

# Quantization


# Lora