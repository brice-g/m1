# FastIA - Classification de Demandes Clients

Ce projet porte sur le fine-tuning d'un modèle de langage (Llama 3.2 1B) pour automatiser la classification des demandes entrantes de la société FastIA.

## 1. Description de l'action corrective et justification
Suite au diagnostic du **Run 1**, plusieurs lacunes ont été identifiées : un écart significatif entre la train_loss et la eval_loss (1.1636), une incapacité à générer systématiquement un JSON valide, et des difficultés de classification sur l'ensemble des catégories.

L'action corrective choisie combine deux approches :

- **Enrichissement du dataset :** Ajout de 39 nouveaux exemples ciblés (passant de 86 à 125 exemples au total) pour couvrir des cas plus variés, ambigus et renforcer les catégories problématiques.

- **Optimisation des hyperparamètres :** Le nombre d'époques d'entraînement a été augmenté de 3 à 4 pour permettre une convergence plus profonde du modèle vers les exemples fournis.

- **Amélioration du Prompting :** Le prompt d'instruction a été affiné pour mieux guider le modèle vers la structure JSON attendue et restreindre les valeurs de sortie aux catégories et priorités autorisées.


## 2. Tableau comparatif Run 1 vs Run 2
| Métrique / Critère | Run 1 (Initial) | Run 2 (Amélioré) |
| :--- | :--- | :--- |
| Taille du dataset	| 86 exemples | 125 exemples (86 + 39) |
| Nombre d'epochs | 3 epochs | 4 epochs |
Train Loss | 1.7566 | 1.1112
Eval Loss | 0.5931 | 0.3169 |
Validité JSON | Non systématique | Systématique (Format respecté) |
Classification | Erreurs fréquentes | Précision accrue |


## 3. Justification du choix du meilleur modèle
Le modèle retenu est celui issu du Run 2 (chargé depuis ./model_final/run2).
Ce choix se justifie par :

- Une meilleure généralisation grâce à l'augmentation du dataset.

- Une conformité stricte au format de sortie attendu (JSON), indispensable pour l'intégration automatique dans l'API.

- de meilleures prédictions sur les catégories et priorités validées lors de l'évaluation qualitative.

## 4. Procédure de lancement de l'API
L'API est construite avec FastAPI et utilise les poids LoRA sauvegardés à l'issue du Run 2.
Prérequis

- Python 3.11

- GPU avec CUDA recommandé (pour de meilleures performances)

1. Installez les dépendances : `pip install fastapi uvicorn torch transformers peft`
2. Lancez le serveur : `python main.py`
3. Testez via interface swagger : `http://localhost:8000/docs`

Exemple de requête POST sur /predict :

`JSON`
`{`
   ` "text": "Bonjour, je n'arrive pas à me connecter à mon tableau de bord depuis ce matin."`
`}`