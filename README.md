# llm-from-scratch
building an LLM from scratch

inspired by
https://youtu.be/zduSFxRajkE

# Articles
https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/


# Installation python
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
# Ubuntu
In order to use the graphviz module in Ubuntu, the graphviz package must be installed
```sudo apt-get install graphviz```

# Structure
```
llm_from_scratch/                   # Root project directory
├── src/                           # Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenizer.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── attention.py
│   │   ├── embedding.py
│   │   ├── transformer.py
│   │   └── llm.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   ├── loss.py
│   │   └── backprop/
│   │       ├── __init__.py
│   │       ├── forward.py
│   │       ├── backward.py
│   │       └── optimizer.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── metrics.py
│
├── data/                          # Training data
│   ├── raw/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── processed/
│       ├── train/
│       ├── valid/
│       └── test/
│
├── docs/                         # Documentation root
│   ├── stylesheets/              # Custom CSS for docs
│   │   └── extra.css
│   ├── javascripts/              # Custom JS for docs
│   │   └── mathjax.js
│   ├── index.md                  # Documentation homepage
│   ├── concepts/                 # Concept documentation
│   │   ├── attention/
│   │   │   ├── attention.md
│   │   │   └── images/
│   │   │       ├── attention_mechanism.png
│   │   │       └── self_attention.png
│   │   ├── transformers/
│   │   │   ├── transformers.md
│   │   │   └── images/
│   │   │       ├── architecture.png
│   │   │       └── block_diagram.png
│   │   └── backprop/
│   │       ├── backprop.md
│   │       └── images/
│   │           ├── computational_graph.png
│   │           └── gradient_flow.png
│   ├── tutorials/
│   │   └── getting_started/
│   │       ├── getting_started.md
│   │       └── images/
│   │           └── setup.png
│   └── api/
│       └── reference.md
│
├── tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_tokenizer.py
│   ├── test_model.py
│   └── test_training.py
│
├── notebooks/                     # Jupyter notebooks
│   ├── data_exploration.ipynb
│   └── model_evaluation.ipynb
│
├── configs/                       # Configuration files
│   ├── model_config.yaml
│   └── training_config.yaml
│
├── scripts/                       # Training scripts
│   ├── train.py
│   └── generate.py
│
├── mkdocs.yml                    # MkDocs configuration (in root)
├── docs-requirements.txt         # Documentation dependencies
├── requirements.txt              # Project dependencies
├── setup.py                      # Package installation
├── README.md                     # Project README
└── .gitignore                    # Git ignore file
```

# Mkdocs
Start development server ```mkdocs serve```
Generate documentation ```mkdocs build```