# -*- coding: utf-8 -*-

import argparse
import numpy as np
import solution_classifieur_lineaire as solution
import gestion_donnees as gd

#################################################
# Execution en tant que script dans un terminal
#################################################


example = '''Exemples:

    (choose from 'gen', 'perceptron', 'sklearn')

    python classifieur_lineaire.py gen 1000 1000 3 0.1
    python classifieur_lineaire.py sklearn 1000 1000 3 0.1
    python classifieur_lineaire.py perceptron 1000 1000 3 0.1
    
    python classifieur_lineaire.py gen 280 280 1 0.5 --donnees_aberrantes
    python classifieur_lineaire.py perceptron 1000 1000 3 0.1 --donnees_aberrantes
'''


def get_arguments():

    parser = argparse.ArgumentParser(
        description='Régression polynomiale.',
        epilog=example,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('methode', choices=['gen', 'perceptron', 'sklearn'],
                        default='gen',
                        help='Choix de la fonction générant les données.')
    parser.add_argument('nb_train', type=int, default=280,
                        help='Nombre de données d\'entraînement.')
    parser.add_argument('nb_test', type=int, default=280,
                        help='Nombre de données de test.')
    parser.add_argument('bruit', type=float, default=0.3,
                        help='Multiplicateur de la matrice de '
                        'variance-covariance (entre 0.1 et 50).')
    parser.add_argument('lamb', type=float, default=0.001,
                        help='Coefficient de régularisation (lambda).')
    parser.add_argument('--donnees_aberrantes', action='store_true',
                        help='Production de données aberrantes.')
    return parser


def main():

    p = get_arguments()
    args = p.parse_args()

    method = args.methode
    nb_train = args.nb_train
    nb_test = args.nb_test
    bruit = args.bruit
    lamb = args.lamb
    donnees_aberrantes = args.donnees_aberrantes

    print("Generation des données d'entrainement...")

    gestionnaire_donnees = gd.GestionDonnees(
        donnees_aberrantes, nb_train, nb_test, bruit)
    [x_train, t_train, x_test, t_test] = gestionnaire_donnees.generer_donnees()

    classifieur = solution.ClassifieurLineaire(lamb, method)

    # Entraînement de la classification linéaire
    classifieur.entrainement(x_train, t_train)

    # Prédictions sur les ensembles d'entraînement et de test
    predictions_entrainement = classifieur.prediction(x_train)
    print("Erreur d'entrainement = ", 100 *
          np.mean(classifieur.erreur(t_train, predictions_entrainement)), "%")

    predictions_test = classifieur.prediction(x_test)
    print("Erreur de test = ", 100 *
          np.mean(classifieur.erreur(t_test, predictions_test)), "%")

    # Affichage
    classifieur.afficher_donnees_et_modele(x_train, t_train, x_test, t_test)


if __name__ == "__main__":
    main()
