# Segmentation d'images par Watershed et U-Net

**Master 2 IA | FSSM | Module Computer Vision | 2026-2027**

**Réalisé par :** Chaouay Lamiae, Midari Aoulia, Rifafi Noura
**Encadré par :** Prof. Qaffou

---

## Présentation

Ce projet porte sur la **segmentation et le comptage de 24 pièces de monnaie en contact** sur une même image. Le notebook `Segmentation_Watershed_UNet_vf.ipynb` se compose de deux parties :

1. **Partie 1 : Watershed contrôlé par marqueurs (OpenCV).** Les marqueurs sont construits à la main (Otsu, morphologie, transformée de distance). Le résultat est comparé au Watershed classique pour mettre en évidence la **sur-segmentation**.
2. **Partie 2 : Watershed + U-Net (PyTorch).** Les étapes manuelles de fabrication des marqueurs sont remplacées par un réseau U-Net qui apprend à les prédire. Watershed reste chargé de tracer les frontières finales.

## Image utilisée

`water_coins.jpg`, tirée du tutoriel officiel OpenCV. Le notebook la télécharge automatiquement depuis `https://docs.opencv.org/4.x/water_coins.jpg` (connexion internet requise à la première exécution).

---

## Contenu du notebook

### Partie 1 : Watershed classique et à marqueurs

| Étape | Description |
|---|---|
| 1 | Chargement de l'image et conversion en niveaux de gris |
| 2 | Seuillage d'Otsu : le seuillage seul ne sépare pas les pièces qui se touchent |
| 3 | Ouverture morphologique et obtention du **fond sûr** (dilatation) |
| 4 | Transformée de distance et **premier plan sûr** (seuil à 70 % du maximum) : 24 noyaux |
| 5 | **Zone inconnue** = fond sûr dilaté − premier plan sûr |
| 6 | Étiquetage des marqueurs (0 = inconnu, 1 = fond, 2 à 25 = pièces) |
| 7 | Application de `cv2.watershed` : 24 pièces séparées, lignes de partage à `-1` |
| 8 | Comparaison de 4 approches : gradient brut, gradient lissé, h-minima, marqueurs |

### Partie 2 : Watershed guidé par U-Net

| Étape | Description |
|---|---|
| 9 | Préparation des cibles (masque + lignes de séparation épaissies) et découpage entraînement (65 % gauche) / test (35 % droite) |
| 10 | Patchs aléatoires 96×96 et augmentation de données (rotations, symétries, luminosité, bruit, flou) |
| 11 | Architecture U-Net compacte (3 niveaux, 16 filtres de base, skip connections) |
| 12 | Entraînement : perte BCE + Dice, Adam (lr 1e-3, décroissance cosinus), 800 itérations |
| 13 | Prédiction sur l'image entière : cartes de probabilité masque et séparation |
| 14 | Watershed guidé par le U-Net (marqueurs = masque − séparation) |
| 15 | Évaluation : nombre de pièces, Dice, IoU par pièce |
| 16 | Test de robustesse sur image dégradée (assombrie, bruitée, floue) |

---

## Principe de la méthode

### Watershed à marqueurs

Watershed simule une inondation à partir de points de départ (les marqueurs). Lorsque deux inondations issues de sources différentes se rencontrent, une ligne de partage est tracée. Fournir soi-même les marqueurs évite la sur-segmentation obtenue lorsque chaque minimum local du gradient devient une source.

### Intégration du U-Net

```
Image ──► U-Net ──► masque + séparation ──► noyaux (marqueurs) ──► cv2.watershed ──► pièces numérotées
```

Le U-Net produit deux canaux : la probabilité d'appartenance à une pièce (**masque**) et la probabilité d'être sur une ligne de partage (**séparation**). Les marqueurs sont obtenus par `masque − séparation`.

### Précaution sur les données

Il n'existe **pas de vérité terrain annotée à la main**. Les étiquettes d'entraînement sont des **pseudo-étiquettes** issues du Watershed à marqueurs de la partie 1. Le U-Net ne peut donc pas, en principe, dépasser sa référence sur la zone d'entraînement. Son intérêt se mesure sur la **zone de test jamais vue** et sur l'**image dégradée**.

---

## Prérequis

- Python 3.8 ou supérieur
- Bibliothèques :

| Bibliothèque | Usage |
|---|---|
| `opencv-python` | Traitement d'image, Watershed |
| `numpy` | Calcul matriciel |
| `matplotlib` | Visualisation |
| `scikit-image` | Gradient de Sobel, h-minima, Watershed classique |
| `torch` (PyTorch) | U-Net et entraînement |
| `jupyter` | Exécution du notebook |

## Installation

```bash
pip install opencv-python numpy matplotlib scikit-image torch jupyter
```

## Exécution

```bash
jupyter notebook Segmentation_Watershed_UNet_vf.ipynb
```

Exécuter ensuite les cellules **dans l'ordre**, de haut en bas : les étapes sont dépendantes les unes des autres (les marqueurs de l'étape 7 servent de référence pour toute la partie 2).

Le notebook détecte automatiquement un GPU (`cuda`) et bascule sur le CPU sinon. L'entraînement (800 itérations sur de petits patchs) reste raisonnable sur CPU.

> Les graines aléatoires sont fixées (`seed = 0`), mais les résultats du U-Net peuvent légèrement varier d'un entraînement à l'autre, notamment selon le matériel utilisé.

---

## Résultats attendus

- **Partie 1 :** 24 pièces correctement séparées avec Watershed à marqueurs, contre plusieurs centaines de régions avec le Watershed classique sur gradient brut (sur-segmentation).
- **Partie 2 :** le U-Net reproduit la segmentation de référence sur la zone d'entraînement ; la comparaison avec la chaîne classique sur zone de test et sur image dégradée est affichée aux étapes 15 et 16. Les valeurs exactes dépendent de l'entraînement : se référer aux sorties du notebook.

## Limites

- **Une seule image** et 24 pièces, dont seulement 6 dans la zone de test : les résultats ne prouvent pas une généralisation à d'autres scènes.
- **Pseudo-étiquettes** issues du Watershed et non annotations manuelles.
- Réglages manuels de la partie 1 (Otsu, noyau 3×3, seuil de 70 %) valables pour cette image mais sensibles à l'éclairage, au bruit et à la taille des pièces.

## Pistes d'amélioration

- Annoter manuellement plusieurs images pour constituer une vraie vérité terrain.
- Ajouter un canal « distance au bord » comme cible supplémentaire du U-Net.
- Comparer avec d'autres approches : Mask R-CNN, Cellpose, StarDist.

---

## Structure du dépôt

```
.
├── Segmentation_Watershed_UNet_vf.ipynb   # Notebook principal
├── water_coins.jpg                        # Image (téléchargée automatiquement)
└── README.md
```

## Références

- Documentation OpenCV : [Image Segmentation with Watershed Algorithm](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html)
- Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- Meyer, F. (1994). *Topographic distance and watershed lines.* Signal Processing.
