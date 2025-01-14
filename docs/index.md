# LLM from scratch
La construction d’un LLM se fait en plusieurs étapes

- collecte des données: téléchargement, suppression des doublons, nettoyages des caractères
- data loader : découpage des données en batch qui peuvent être traités
- tokenization : mise en place d’une tokenizer pour transformer le texte en tokens
- embedding : embedding des tokens dans un espace vectoriel
- attention : mécanisme d’attention pour ajouter du contexte aux embeddings de mots