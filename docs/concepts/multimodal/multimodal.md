https://www.youtube.com/watch?v=vAmKB7iPkWw

# constrative learning
On a une série d’images avec leur description associée.
on a un encoder d’images qui prend une image et génère un embedding de cette image Ei
on a un encodeur de texte qui prend la description et génère un embedding Et
On veut maximiser la correspondance entre l’embedding d’une image et l’embedding de sa description.

Pour cela on mixe différentes images et leurs descriptions et on veut maximiser le produit scalaire entre Ei.Et quand c’est la bonne paire image/description.
C’est le principe de la cross entropie.

pseudo code
# extract feature representation for each modalilty
I_f = image_encoder(I) # [n,d_i] - we have n images, d_i : size of embedding
T_f = text_encoder(T)  # [n,d_t] - we have n description, d_t : size of text embedding

# joint embedding [n,d_e]
# Wi [d_i,d_e] : learned projection 
# Wt [d_t,d_e] : learned projection
I_e = l2_normalize(np.dot(I_f, Wi),axis=1) #  project to same dimension (d_e) and normalize [n,d_e]
T_e = l2_normalize(np.dot(T_f, Wt),axis=1) # project to same dimension (d_e) and normalize

# temp : learned temperature
logits = np.dot(I_e,T_e.T)* np.exp(temp) # [n,n]

# symetric loss function
# we want to maximize the diagonal (same index for image and description)
labels = np.arange(n) [ 1,n]
loss_i = cross_entropy(logits,labels,axis=0)
loss_t = cross_entropy(loigts, labels,axis=1)
loss = (loss_i+loss_t)/2

# clip - cross-entropy
Pour calculer la fonction de perte on doit faire 
softmax sur toutes les lignes
softmax sur toutes les colonnes.

Or on a un risque avec le softmax que les valeurs soient trop grandes et qu’on soit en overflow (nombre trop grand pour être représenté).
En effet le softmax utilise la fonction exponentielle qui grimpe très vite.
On peut normaliser, pour cela et manipuler le softmax de façon à réduire sa valeur. On veut au final une probabilité, on peut multiplier la fraction par une constante au numérateur et dénominateur, sans changer la valeur de la fraction
softmax(ai) = e(ai) / sum(e(ak)) 
= c * e(ai) / (c * sum(e(ak)))
= e(log(c)) * e(ai) / (e(log(c)) * sum(e(ak)))
= e(ai+log(c)) / sum(e(ak+log(c)))

Il nous suffit de prendre une constance c qui va empêcher l’expontielle de prendre des valeurs ttrop importante. Pour cela, on prend c = -log(max(ai))
Le problème de cette approche c’est :
- on a beacoup de calculs à faire , des exp, des logs, et trouver un max sur chaque ligne et sur chaque colonne
- trouver un max implique que si on veut paralléliser il faut envoyer toute la ligne ou toute la colonne
# sigclip
Pour résoudre le problème de l’exponentielle, au lieu de considérer chaque ligne (respectivement chaque colonne) comme un vecteur dont veut s’assurer qu’il est de la forme [1,0,0,0,0 ...] on va considérer que chaque case de notre matrice est un problème de classification binaire.
On veut que pour chaque entrée dans la diagonale, le modèle prédise un 1, et pour tous les autres produits scalaires que le modèle prédise un 0.
Cela revient à traiter chaque produit scalaire indépendamment.
Classification binaire, on utilse la fonction sigmoid.

# 
