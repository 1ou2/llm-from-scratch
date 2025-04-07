# Diffusion model
Le principe est de partir d’une image faite de bruit (des pixels aléatoires) et d’arriver à générer une image cohérente.
Pour cela on va entrainer un modèle à enlever le bruit d’une image.
On part d’une image (nette !) et on construit une série d’images avec de plus en plus de bruits.

# U-net
https://www.youtube.com/watch?v=sFztPP9qPRc
on part d’une image, on application un réseau de convolution pour extraire des caractéristiques.
Puis on dégrade la qualité de l’image (on baisse la résolution), si on garde la même taille de kernel, alors c’est comme si on avait fait un zoom-out. Le kernel couvre une plus grande partie de l’image.

