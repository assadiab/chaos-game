#La matrice du Chaos Game représente les occurrences de chaque k-merdans la séquence d'ADN,
# tandis que la matrice de Signature Génomique représente la composition en mots de deux lettres
# (di-nucléotides) dans la séquence.

#La matrice du Chaos Game est une représentation visuelle de la distribution des k-mers dans la séquence d'ADN,
#tandis que la matrice de Signature Génomique fournit une vue plus abstraite de la séquence,
# en montrant les motifs récurrents de di-nucléotides.

import numpy as np
import matplotlib.pyplot as plt
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
    ---------
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


def generate_chaos_game_representation(seq: str, k: int, size: int) -> np.ndarray:
    """
    Génère une représentation du Chaos Game à partir d'une séquence d'ADN.

    Paramètres
    ----------
    seq : str
        Séquence d'ADN à partir de laquelle générer la représentation du Chaos Game.
    k : int
        Longueur des k-mers à considérer.
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
    section_size = size // 2**k  # Taille de chaque section
    
    # Parcours de la séquence pour générer les points
    current_position = np.array([0.5, 0.5])  # Position initiale au centre du carré

    for i in range(0, len(seq) - k + 1, k):
        kmer = seq[i:i+k]  # Extraction du k-mer à partir de la séquence
        next_position = nucleotide_coords.get(kmer[-1], np.array([0.5, 0.5]))  # Utilisation de get pour gérer les nucléotides inconnus

        # Mise à jour de la position actuelle
        current_position = (current_position + next_position) / 2

        # Redimensionnement des coordonnées pour qu'elles s'inscrivent dans les limites de l'image
        x = min(int(current_position[0] * section_size), section_size - 1)  # Coordonnée x avec une limite de taille
        y = min(int(current_position[1] * section_size), section_size - 1)  # Coordonnée y avec une limite de taille

        # Attribution du k-mer à la section correspondante
        chaos_game_representation[y, x] += 1  # Incrémentation du compteur du k-mer

    return chaos_game_representation  # Retourne la matrice de représentation du Chaos Game


def generate_signature_genome(seq: str, word_length: int) -> np.ndarray:
    """
    Génère une matrice de signature génomique à partir d'une séquence d'ADN et d'une longueur de mot spécifiée.

    Paramètres
    ----------
    seq : str
        Séquence d'ADN.
    word_length : int
        Longueur du mot.

    Renvois
    -------
    signature_genome : np.ndarray
        Matrice de signature génomique.
    """
    words = [seq[i:i+word_length] for i in range(len(seq) - word_length + 1)]
    alphabet = ['A', 'T', 'C', 'G']
    signature_genome = np.zeros((4**word_length, 4))
    for i, word in enumerate(words):
        row = int(''.join([str(alphabet.index(letter)) for letter in word]), 4)
        column = alphabet.index(word[-1])
        signature_genome[row, column] += 1
    return signature_genome / np.sum(signature_genome, axis=1, keepdims=True)


def write_matrix_to_file(matrix: np.ndarray, filename: str):
    """
    Écrit une matrice dans un fichier.

    Paramètres
    ----------
    matrix : np.ndarray
        Matrice à écrire dans le fichier.
    filename : str
        Nom du fichier de sortie.
    """
    np.savetxt(filename, matrix, delimiter=',')  # Écriture de la matrice dans le fichier avec les valeurs séparées par des virgules


def read_genbank_sequence(genbank_file):
    """
    Lit le fichier GenBank et renvoie la séquence d'ADN.
    """
    for record in SeqIO.parse(genbank_file, "genbank"):
        seq = str(record.seq)
        return seq


# Exemple d'utilisation
if __name__ == "__main__":
    genbank_file = "NC_001133.gbk"
    dna_sequence = read_genbank_sequence(genbank_file)
    kmer_length = 3  # Longueur des k-mers à considérer
    image_size = 1000  # Taille de l'image de sortie en pixels

    # Génération de la représentation du Chaos Game
    chaos_game_matrix = generate_chaos_game_representation(dna_sequence, kmer_length, image_size)

    # Génération de la signature génomique
    word_length = 2  # Longueur du mot
    signature_genome = generate_signature_genome(dna_sequence, word_length)

    # Affichage de la matrice du Chaos Game
    print("Matrice du Chaos Game :")
    print(chaos_game_matrix)

    # Affichage de la matrice de signature génomique
    print("Matrice de Signature Génomique :")
    print(signature_genome)

    # Écriture des matrices dans des fichiers
    write_matrix_to_file(chaos_game_matrix, "chaos_game_matrix.csv")
    write_matrix_to_file(signature_genome, "signature_genome_matrix.csv")

    # Affichage des matrices avec Matplotlib
    plt.imshow(chaos_game_matrix, cmap='gray', origin='upper')
    plt.title("Représentation du Chaos Game")
    plt.colorbar()
    plt.show()

    plt.imshow(signature_genome, cmap='hot', origin='upper')
    plt.title("Matrice de Signature Génomique")
    plt.colorbar()
    plt.show()

