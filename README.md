# Projet-Machine_Learning
Projet de Reconnaissance de Lettres Manuscrites avec MLP Ce projet consiste à dessiner une lettre sur une interface graphique (Tkinter + CustomTkinter), extraire les caractéristiques visuelles sous forme de coefficients, et utiliser un réseau de neurones (MLP) entraîné pour prédire la lettre manuscrite.

Objectif:

Créer un outil capable de reconnaître une lettre manuscrite dessinée à la main sur un canvas, en se basant sur une grille d’analyse et un réseau de neurones Multi-Layer Perceptron (MLP) entraîné sur des coefficients extraits.

Lancer neurone_network.py
➤ Entraîne le modèle de réseau de neurones sur un dataset (all_coef.txt) contenant les coefficients des lettres manuscrites.
➤ Le modèle est sauvegardé sous le nom mlp_model_3couches.pkl.
➤ Affiche un bilan : précision globale, précision par lettre, matrice de confusion.

Lancer interface.py
➤ Ouvre une interface graphique où l’utilisateur peut dessiner une lettre, puis cliquer sur "Scan" pour que le modèle la reconnaisse.
➤ Les boutons "Effacer" et "Enregistrer" permettent de réinitialiser ou sauvegarder le dessin.
➤ Le résultat de la reconnaissance s’affiche directement sous forme de lettre.

Les lettres supportées
➤ Le système reconnaît les lettres : B, F, G, H, J, K, chacune encodée en binaire sur 3 bits.

Fichier all_coef.txt
➤ Contient une ligne par image, avec 128 coefficients + 3 bits (le code binaire de la lettre).
➤ Ce fichier sert de base d’apprentissage pour le modèle MLP.
