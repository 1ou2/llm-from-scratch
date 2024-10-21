# Git
```
# create a new feature branch
git checkout -b new-feature
# add changes
git add xx
git commit
# push to new branch
git push -u origin new-feature
# push to main branch
git checkout main
git pull
git pull origin new-feature
# resolve conflicts
git push
```

# Micrograd
Bibliothèque python permenttant d’illustrer et de calculer des gradients
- Encapsulation des valeurs dans des objets Value
- opération +,-,*,/ et puissance (**)
- permet de calculer la backpropagation
- fonction backward() : initie le calcul de backpropagation, on va calculer toutes les dérivées des expressions, qui ont amenées à cette valeur. Ex: si g dépend de a,b,c,d,e,f, avec a et b valeurs d’entrées et c,d,e, et f des valeurs intermédiaires on va pouvoir calculer la dérivée de g par rapport à a et donc savoir comment on doit faire évoluer a pour que g varie dans le "bon sens". Si on veut diminuer g, on saura comment modifier a et b pour diminuer g

Exemple :
Si dg/da = 130 alors une petite augmentation de a correspond à une augmentation avec une pente de 130