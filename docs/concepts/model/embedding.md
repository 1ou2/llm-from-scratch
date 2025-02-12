# Embedding
L’embedding d’un mot, d’une phrase est la représentation vectorielle de ce mot ou de cette phrase.
Les LLM utilisent des espaces vectoriels de grande dimension pour représenter des concepts.

Un embedding est une table qui associe à chaque token une représentation vectorielle. Si on a un `vocabulaire` de 50256 tokens (c’est le cas dans GPT-3), et qu’on projette dans un espace vectoriel de dimension 12288, on crée alors une table de dimension 50256x2048.
```python
    vocab_size = 50256 # GPT-3 vocab size
    output_dim = 12288 # GPT-3 embedding size
    # create a random embedding matrix
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
``` 
Comme c’est juste une table de référence, on n’a pas l’information de la position du token dans la phrase. Pourtant, c’est important ! Dans la phrase "Alice a tué Bob" ne veut pas dire la même chose que "Bob a tué Alice". Et pourtant, les mot Bob va avoir la même représentation dans ces phrases. Pour capturer cette information, on va modifier l’embedding des mots en ajoutant la position.
Il y a plusieurs méthodes pour ajouter cette information au vecteur d’embedding.
Ce qu’on peut faire c’est créer une nouvelle table d’embedding correspondant cette fois à la position dans la phrase. Pour cela on a besoin de connaitre la taille du contexte, et pour chaque position on associe un vecteur.
```python
    vocab_size = 50256 # GPT-3 vocab size
    output_dim = 12288 # embedding size
    context_length = 2048 # context length
    
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

```

Ensuite pour obtenir un embedding intégrant la position du mot dans la pharse il suffit d’ajouter ces deux embeddings.
Note : GPT-3 utilise une technique plus complexe avec des sinusoïdes.
[![](images/embedding.png)](images/embedding.png)
```python
    vocab_size = 50256 # GPT-3 vocab size
    output_dim = 12288 # embedding size
    context_length = 2048 # context length

    # create a random embedding matrix
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    # position embedding
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    

    x,y = next(iter(dataloader))
    token_embedding = token_embedding_layer(x)
    pos_embddings = pos_embedding_layer(torch.arange(context_length))
    input_embedding = token_embedding + pos_embddings
```