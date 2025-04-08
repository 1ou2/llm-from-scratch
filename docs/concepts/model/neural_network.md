# MLP
C’est un réseau fully connected.
Dans les LLM, on a souvent

def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states,approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states,approximate="tanh")
        return hidden_states

On a la première couche qui va typiquement ajouter des paramètres on va passer d’une taille 512 à 2048 par exemple, puis la deuxième couche qui va revenir à la taille initiale.
Les réseaux de neurones :
- ajoute des paramètres au réseau en plus des blocs transformers, et donc permettent de donner plus de complexité au modèle
- ajoute de la non-linéarité, permettant de modéliser des phénonèmes plus complexes
- permette au réseau de s’ajuster entre les blocs de transformers, en changeant le flot d’information et en adaptant les données pour la couche suivante

# Non-linéarité
En théorie toute fonction non linéraire fait le job. Dans la pratique, le réseau va converger ou pas et le choix de la fonction de non linéarité influe, car cela change la façon dont le gradient est propagé.
Par exemple avec Relu, toute valeur négative devient 0, et comme on multiplie par cette valeur, ça rend le gradient nul, donc le réseau ne peut pas apprendre 