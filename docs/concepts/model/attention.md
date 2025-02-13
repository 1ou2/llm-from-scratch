# introduction
L’attention est le mécanisme qui permet à un réseau de neurones de donner du poids, de porter de l’attention à certaines parties de la phrase.
1. "Alice a mangé une pizza et elle était délicieuse"
2. "Alice a mangé une pizza et elle était contente"
Dans la phrase 1 le mot elle fait référence a la pizza, alors que dans la phrase 2, le mot elle fait référence à Alice. Pour transmettre ce type de subtilités on a inventé le concept d’attention. Le mot "elle" va porter attention à "pizza" dans la phrase 1, alors que l’attention portera sur Alice dans la phrase 2

# Self attention
Il s’agit de chercher la simalirité entre chaque mot de la phrase et tous les autres mots (incluant lui-même).
Une fois cette similarité calculée, elle va servir à savoir comment encoder chaque mot.
[![](images/attention-formula.png)](images/attention-formula.png)

Les paramètres sont :
- Q : query
- K : key
- V : value

Ce vocabulaire est issu des bases de données. La `query` est le terme recherché, `key` est la clé de recherche qui est la plus pertinente par rapport à notre terme recherché. Enfin `value` est la valeur retournée par cette recherche.

La formule se décompose comme suit :
- Q.K^t : c’est le produit scalaire entre la query et la key. Cela revient a calculer la proximité entre chaque terme.
- Division par racine carrée (dimension key): c’est pour des raisons de scalabilité. L’algorithme va ainsi converger plus rapidement. C’est une étape de normalisation
- Softmax : cela transforme les valeurs obtenues en pourcentage. On obtient pour chaque token de la phrase le pourcentage d’attention de chacun des autres tokens
- on utilise le pourcentage pour savoir quel poids donner à chaque `value`

