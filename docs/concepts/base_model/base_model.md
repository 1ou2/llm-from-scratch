Une fois nos données, collectées, triées et transformées en tokens, on passe à l’étape la plus importante la création d’un Base Model.
C’est cette étape qui est la plus couteuse. On va entrainer un système GPT sur une quantité massive d’informations.
Mais avant de parler de l’entrainement. Qu’est-ce que GPT.

# Entrées et sortie
C’est un système qui prend en entrée des nombres appelés aussi tokens et qui en sortie génère le prochain token
[![](images/llm-in-out.png)](images/llm-in-out.png)

# LLM
Si on zoom sur le LLM il peut être décomposé en deux parties :
- GPT : generative pre-trained transformer
- Selection du prochain token
[![](images/gpt-select-token.png)](images/gpt-select-token.png)

# GPT


[![](images/gpt.png)](images/gpt.png)

## Transformer
Comme son nom l'indique, le composant principal de GPT est le bloc transformer. On a une séquence de blocs transformer qui s'enchainent
[![](images/transformer.png)](images/transformer.png)

### Feed Forward

# Sélection du prochain token
La sortie du bloc gpt est un tensor contenant ce qu'on appelle des logits. Ce sont les scores attribués par le réseau de neurones.
La haute parallélisation de l'architecture des transformers fait que chaque token d'entrée est traité en parallèle. On obtient donc en sortie du bloc GPT `context_length` logits.
Chaque logits est un tensor de dimension `vocab_size`. On a un score associé à chaque token de notre vocabulaire.
Supposons qu'on a 3 tokens en entrée t0, t1 et t2.
Appellons l0, l1 et l2 les logits calculés (en parallèle) pour chacun de ces tokens.
- l0 : représente les scores prédits par le réseau de neurone pour le token qui suivra le token 0, c'est donc les scores du token 1
- l1 : représente les scores prédits par le réseau de neurone pour le token qui suivra le token 1, c'est donc les scores du token 2
- l2 : représente les scores prédits par le réseau de neurone pour le token qui suivra le token 2, c'est donc les scores du token 3
Or nous connaissons déjà les valeurs des tokens 0, 1 et 2 puisque ce sont nos tokens d'entrée. Connaitre les scores de prédictions l0 et l1 ne nous sert à rien.
Ce qui nous intéresse c'est l2, car il contient prédictions pour le token 3.

Note le fait de calculer l0 l1 et l2 alors qu'on ne va utiliser que l2 est intrinsique à l'architecture des transformers :
- tout est calculé en parallèle
- on doit faire tous ces calculs car le mécanisme d'attention a besoin de voir toute la séquence de tokens pour faire sa prédiction

[![](images/select-token.png)](images/select-token.png)

À partir des scores (logits) attribués par le réseau à chaque token possible de notre vocabulaire on peut en déduire une probabilité. Pour cela on utilise la fonction `softmax`.
Ensuite, soit on a une stratégie dite `greedy`, qui consiste à choisir systématiquement le token le plus probable (à ce moment là on aurait pu éviter le softmax et prendre juste le token du vocabulaire qui a le score le plus élevé), soit on échantillone à partir de cette distribution pour choisir le prochain token.

```python
def generate_next_token(model, context):
    # context shape: (1, sequence_length)
    # logits shape: (1, sequence_length, vocab_size)
    logits = model(context)
    
    # Get only the last position's predictions
    # last_logits shape: (1, vocab_size)
    last_logits = logits[:, -1, :]
    
    # Apply softmax to convert to probabilities
    probs = F.softmax(last_logits, dim=-1)
    
    # Sample from the probability distribution
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token
```

## Échantillonnage
Échantillonner à partir de la distribution veut dire qu'on va tirer aléatoirement le prochain token (ce n'est pas forcément le plus probable qui va être choisi), mais ce n'est pas non plus du hasard. Si un token t0 a probalité de  50%, un autre t1 de 20% et tous les autres sont à moins de 1%, on aura une chance sur deux de tirer t0 et une chance sur cinq de tirer t1

## Logits
Le terme logits est une convention de nommage issue des mathématiques.
En mathématique la fonction logit tire son nom du logarithme:
- c'est la fonction inverse de sigmoid
- logit(p) = log(p/(1-p))
- Cela associe des probabilités (entre 0 et 1) à des nombres dans l'intervale[-∞, +∞]
En deep learning, on n'utilise pas la fonction `sigmoid` car on ne fait une classification binaire. On utilise `softmax`.
Par abus de langage le terme `logits` est utilisé mais cette fois pour representer le score brut donné par le réseau
# Vison globale

[![](images/llm-full.png)](images/llm-full.png)