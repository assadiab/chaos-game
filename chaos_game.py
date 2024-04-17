"""
Nom du fichier : chaos_game.py
Description : Ce fichier contient le code source l'implémentation de l'algorithme du Chaos Game pour générer une représentation visuelle d'une séquence d'ADN.
Auteur : Assa DIABIRA & Inès MANOUR
Dernière modification : 17/04/2024
"""

import numpy as np  # Importation de la bibliothèque NumPy pour le calcul numérique
import matplotlib.pyplot as plt  # Importation de la bibliothèque Matplotlib pour la création de graphiques
from Bio import SeqIO

# Définition des coordonnées des bases ATCG dans le carré
nucleotide_coords = {
    'A': np.array([0, 0]),  # Coordonnées pour la base A
    'T': np.array([1, 0]),  # Coordonnées pour la base T
    'C': np.array([0, 1]),  # Coordonnées pour la base C
    'G': np.array([1, 1])   # Coordonnées pour la base G
}

def count_kmers(seq: str, k: int) -> dict:
    """
    Compte le nombre d'occurrences de chaque k-mer dans la séquence.

    Paramètres
    ----------
    seq : str
        Séquence nucléotidique.
    k : int
        Longueur des k-mers à considérer.

    Renvois
    -------
    kmer_count : dict
        Dictionnaire contenant les k-mers comme clés et leur nombre d'occurrences comme valeurs.
    """
    kmer_count = {}  # Initialisation du dictionnaire de comptage des k-mers
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]  # Extraction du k-mer à partir de la séquence
        if "N" not in kmer:  # Ignorer les k-mers contenant des 'N'
            kmer_count[kmer] = kmer_count.get(kmer, 0) + 1  # Incrémentation du compteur du k-mer
    return kmer_count  # Retourne le dictionnaire de comptage des k-mers

def probabilities(kmer_count: dict) -> dict:
    """
    Calcule les probabilités pour chaque k-mer.

    Paramètres
    ----------
    kmer_count : dict
        Dictionnaire contenant les comptes de chaque k-mer.

    Renvois
    -------
    kmer_probs : dict
        Dictionnaire contenant les k-mers comme clés et leur probabilité d'occurrence comme valeurs.
    """
    total_count = sum(kmer_count.values())  # Calcul du nombre total de k-mers
    if total_count == 0:
        return {}  # Éviter la division par zéro
    return {kmer: count / total_count for kmer, count in kmer_count.items()}  # Calcul des probabilités

def generate_chaos_game_representation(seq: str, size: int) -> np.ndarray:
    """
    Génère une représentation du Chaos Game à partir d'une séquence d'ADN.

    Paramètres
    ----------
    seq : str
        Séquence d'ADN à partir de laquelle générer la représentation du Chaos Game.
    size : int
        Taille de l'image de sortie (en pixels).

    Renvois
    -------
    chaos_game_representation : np.ndarray
        Matrice représentant le Chaos Game.
    """
    # Initialisation de la matrice de représentation
    chaos_game_representation = np.zeros((size, size))  # Création d'une matrice de zéros
    
    # Division de l'image en sections
    section_size = size // 2  # Taille de chaque section
    
    # Parcours de la séquence pour générer les points
    current_position = np.array([0.5, 0.5])  # Position initiale au centre du carré

    for nucleotide in seq:
        # Calcul des coordonnées du prochain point
        next_position = nucleotide_coords.get(nucleotide, np.array([0.5, 0.5]))  # Utilisation de get pour gérer les nucléotides inconnus

        # Mise à jour de la position actuelle
        current_position = (current_position + next_position) / 2

        # Redimensionnement des coordonnées pour qu'elles s'inscrivent dans les limites de l'image
        x = min(int(current_position[0] * section_size), section_size - 1)  # Coordonnée x avec une limite de taille
        y = min(int(current_position[1] * section_size), section_size - 1)  # Coordonnée y avec une limite de taille

        # Attribution du nucléotide à la section correspondante
        chaos_game_representation[y, x] += 1  # Incrémentation du compteur du nucléotide

    """Affichage de la matrice du Chaos Game (pour le débogage)"""
    print("Matrice du Chaos Game :")
    print(chaos_game_representation)

    

    # Affichage de la matrice avec Matplotlib
    plt.imshow(chaos_game_representation, cmap='gray', origin='upper')
    plt.title("Représentation du Chaos Game")
    plt.colorbar()
    plt.show()
    
    return chaos_game_representation  # Retourne la matrice représentant le Chaos Game


def read_genbank_sequence(genbank_file):
    """
    Lit le fichier GenBank et renvoie la séquence d'ADN.
    """
    for record in SeqIO.parse(genbank_file, "genbank"):
        seq = str(record.seq)
        print("Séquence d'ADN extraite :")
        print(seq)
        return str(record.seq)


# Vérification des coordonnées des nucléotides
for nucleotide, coord in nucleotide_coords.items():
    print(f"Coordonnées de {nucleotide} : {coord}")


# Exemple d'utilisation
if __name__ == "__main__":
    genbank_file = "CR380953.gbk"
    dna_sequence = read_genbank_sequence(genbank_file)
    print("Séquence d'ADN :")
    print(dna_sequence)

    kmer_length = 3  # Longueur des k-mers à considérer
    image_size = 1000  # Taille de l'image de sortie en pixels
   
    kmer_counts = count_kmers(dna_sequence, kmer_length)  # Comptage des k-mers dans la séquence
    kmer_probs = probabilities(kmer_counts)  # Calcul des probabilités des k-mers
    chaos_game_matrix = generate_chaos_game_representation(dna_sequence, image_size)  # Génération de la représentation du Chaos Game
  
"""
#c'etais pr voir si le soucis venait de la sequence,Lit le fichier GenBank et renvoie la séquence d'ADN.
def read_genbank_sequence(genbank_file):
    
    
    for record in SeqIO.parse(genbank_file, "genbank"):
        return str(record.seq)

# Exemple d'utilisation
if __name__ == "__main__":
    genbank_file = "NC_001133.gbk"
    dna_sequence = read_genbank_sequence(genbank_file)
    print("Séquence d'ADN :")
    print(dna_sequence)
"""

