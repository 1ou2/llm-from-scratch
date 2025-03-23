# Longueur de la génération
Quelle que soit la longueur de la séquence d’entrée le modèle génère content_length tokens.
Dans GPT-2 on a un un token <|endoftext|> qui marque la fin des séquences. Mais mêne après qu’il soit généré le modèle génère d’autres tokens. C’est au niveau applicatif qu’on identifie 
ce token et qu’on tronque la réponse.

# Utilisation du mask en inference
Durant l'entrainement on utilise un causal mask qui masque les futurs tokens afin de s'assurer que le modèle génère bien des réponses en fonction des inputs sans tricher et voir la solution. Ce masque est aussi utilisé durant l'inférence. on pourrait penser qu'il n'est pas nécessaire, mais c'est le design du modèle qui fait ça.
Concrètement durant l'inférence si on a les tokens a,b,c et d.
- le token a n'aura que les informations de contexte sur lui-même. C'est à dire que le vecteur de contexte généré par le mécanisme d'attention n'est calculé qu'à partir des informatinos issus de ce token
- b est issu de a et b
- c de a,b et c
- d de a,b, c et d

## Pourquoi ?
Le modèle a été entrainé comme ça, si durant l'inférence on change et on calcule le vecteur de contexte sans le masque, on n'obtiendra des résultats incohérents.

# KV cache
C'est une technique d'optimisation