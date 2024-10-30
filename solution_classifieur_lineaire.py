# -*- coding: utf-8 -*-

#####
# Vos Noms (VosMatricules) .~= À MODIFIER =~.
####

import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt


class ClassifieurLineaire:
    def __init__(self, lamb, methode):
        """
        Algorithmes de classification lineaire

        L'argument ``lamb`` est une constante pour régulariser la magnitude
        des poids w et w_0

        ``methode`` :   1 pour classification generative
                        2 pour Perceptron
                        3 pour Perceptron sklearn
        """
        self.w = np.array([1., 2.])  # paramètre aléatoire
        self.w_0 = -5.               # paramètre aléatoire
        self.lamb = lamb
        self.methode = methode

    def entrainement(self, x_train, t_train):
        """
        Entraîne deux classifieurs sur l'ensemble d'entraînement formé des
        entrées ``x_train`` (un tableau 2D Numpy) et des étiquettes de classe
        cibles ``t_train`` (un tableau 1D Numpy).

        """
        if self.methode == 'gen':
            self.entrainement_generatif(x_train, t_train)

        elif self.methode == 'perceptron':
            self.entrainement_perceptron(x_train, t_train)

        else:
            self.entrainement_sklearn(x_train, t_train)

        print('w = ', self.w, 'w_0 = ', self.w_0, '\n')

    def entrainement_generatif(self, x_train, t_train):
        """
        Implémenter la classification générative de
        la section 4.2.2 du libre de Bishop. Cette méthode doit calculer les
        variables suivantes:

        - ``p`` scalaire spécifié à l'équation 4.73 du livre de Bishop.

        - ``mu_1`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.75 du livre de Bishop.

        - ``mu_2`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.76 du livre de Bishop.

        - ``sigma`` matrice de covariance (tableau Numpy 2D) de taille DxD,
                    telle que spécifiée à l'équation 4.78 du livre de Bishop,
                    mais à laquelle ``self.lamb`` doit être ADDITIONNÉ À LA
                    DIAGONALE (comme à l'équation 3.28).

        NOTE: Ne nécéssite aucune boucle.

        Résultats:

        - ``self.w`` un vecteur (tableau Numpy 1D) de taille D tel que
                    spécifié à l'équation 4.66 du livre de Bishop.

        - ``self.w_0`` un scalaire, tel que spécifié à l'équation 4.67
                    du livre de Bishop.
        """

        # Classification generative
        print('Classification générative.')

        # AJOUTER CODE ICI

        p = np.mean(t_train)
        mu_1 = np.mean(x_train[t_train == 1], axis=0)
        mu_2 = np.mean(x_train[t_train == 0], axis=0)
        
        sigma = (1 - p) * np.cov(x_train[t_train == 1], rowvar=False) + \
                p * np.cov(x_train[t_train == 0], rowvar=False)
        sigma += np.eye(sigma.shape[0]) * self.lamb
        
        sigma_inv = np.linalg.inv(sigma)
        self.w = sigma_inv @ (mu_1 - mu_2)
        self.w_0 = -0.5 * mu_1.T @ sigma_inv @ mu_1 + \
                0.5 * mu_2.T @ sigma_inv @ mu_2 + np.log(p / (1 - p))
        print('Classification générative completed.')

        # La question suivante ne vaut aucun point, mais détermine si vous
        # avez bien compris les modèles génératifs: que se passe-t-il si des
        # données aberrantes se retrouvent dans les données d'entraînement et
        # pourquoi cela se produit-il ?
        # Répondez en commentaire:

        # Fonction de déboggage. Celle-ci permet de visualiser les gaussiennes
        # calculées pour chaque classe. Il vous faudra calculer les variables
        # intermédiaires sigma_1, sigma_2, les matrices de covariances pour
        # chaque classe. METTRE EN COMMENTAIRE DANS LA REMISE.
        # self.afficher_parametres_gaussienne(
        #     x_train, t_train, mu_1, mu_2, sigma_1, sigma_2)

    def entrainement_perceptron(self, x_train, t_train):
        """
        Implémenter la classification par Perceptron tel que de présenté
        dans les notes de cours. L'algorithme doit implémenter la descente
        de gradient **stochastique**.

        La SGD doit utiliser un taux d'apprentissage de 0.001 et un
        nombre d'itérations (epochs) == 1000 (maximum). N'oubliez pas la
        régularisation (self.lambda)!

        NOTE: Nécéssite deux boucles.

        Résultat:

        - ``self.w`` un vecteur (tableau Numpy 1D) de taille D la frontière de
                séparation.

        - ``self.w_0`` le biais.
        """
        print('Classification par Perceptron')
        # AJOUTER CODE ICI

        n_samples, n_features = x_train.shape
        self.w = np.zeros(n_features)
        self.w_0 = 0.0
        learning_rate = 0.001
        n_epochs = 1000

        for epoch in range(n_epochs):
            for i in range(n_samples):
                linear_output = self.prediction(x_train[i])
                y_pred = 1 if linear_output >= 0 else 0
                
                if y_pred != t_train[i]:
                    update = learning_rate * (t_train[i] - y_pred)
                    self.w += update * x_train[i]
                    self.w_0 += update

            self.w -= self.lamb * self.w
        print('Classification par Perceptron completed.')

    def entrainement_sklearn(self, x_train, t_train):
        """
        Utiliser la classe Perceptron de sklearn pour obtenir un classifieur
        binaire. Les paramètres sont les même que pour votre propre
        implémentation, mais vous devez aussi spécifier la fonction de coût
        'l2'. N'oubliez pas la régularisation !

        ATTENTION: sklearn n'utilise pas toujours la même nomenclature que
        le cours et le livre de Bishop. Lisez bien la documentation de la
        classe avant de l'appeler.

        ATTENTION: La classe Perceptron effectue un traitement des données
        supplémentaire si fit_intercept = True. Vous devez le mettre à False.

        NOTE: Votre implémentation du Perceptron et celle de sklearn devraient
        donner des résultats très similaires. Une bonne pratique est donc
        d'implémenter cette version puis de débogger votre propre
        implémentation pour qu'elle donne des résultats similaires.

        Résultats :

        - ``self.w`` un vecteur (tableau Numpy 1D) de taille D la frontière de
                séparation.

        - ``self.w_0`` le biais.
        """

        print('Classification par Perceptron [sklearn]')
        # AJOUTER CODE ICI

        feature_means = x_train.mean(axis=0)
        x_train_fit = x_train - feature_means
        self.w_0 = 0

        perceptron = Perceptron(penalty='l2', alpha=self.lamb, fit_intercept=False)
        perceptron.fit(x_train_fit, t_train)

        self.w = perceptron.coef_.flatten()
        self.w_0 = -np.dot(feature_means, self.w)

        print('Classification par Perceptron [sklearn] completed.')

    def prediction(self, x):
        """
        Retourne la prédiction du classifieur lineaire.  Retourne 1 si x est
        devant la frontière de décision et 0 sinon.

        ``x`` est une seule ou plusieurs données.

        Cette méthode suppose que la méthode ``entrainement()``
        a préalablement été appelée. Elle doit utiliser les champs ``self.w``
        et ``self.w_0`` afin de faire cette classification.

        NOTE: Ne nécéssite aucune boucle et ne devrait pas nécéssiter de
        condition particulière si x représente une seule données ou plusieurs.
        """
        # AJOUTER CODE ICI

        return (np.dot(x, self.w) + self.w_0)

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de classification, i.e.
        1. si la cible ``t`` et la prédiction ``prediction``
        sont différentes, 0. sinon.

        NOTE: Comme au précédent travail, le code devrait être
        le même si t et prédiction sont des vecteurs ou des scalaires.

        NOTE: Ne nécéssite aucune boucle.
        """
        # AJOUTER CODE ICI
        return np.mean(t != prediction)

    def afficher_donnees_et_modele(self, x_train, t_train, x_test, t_test):
        """
        afficher les donnees et le modele

        x_train, t_train : donnees d'entrainement
        x_test, t_test : donnees de test
        """
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1],
                    s=t_train * 100 + 20, c=t_train)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Training data')

        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=t_test * 100 + 20, c=t_test)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Testing data')

        #plt.show()
        
        plt.savefig('Resultat1.png')

    def parametres(self):
        """
        Retourne les paramètres du modèle
        """
        return self.w_0, self.w

    def afficher_parametres_gaussienne(
        self, x_train, t_train, mu1, mu2, sigma1, sigma2
    ):
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1],
                    c=t_train)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_train[:, 0]) - 2, np.max(x_train[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Gaussiennes')

        def plot_contour(mu, sigma):
            # https://stackoverflow.com/a/65992443

            cov_inv = np.linalg.inv(sigma)  # inverse of covariance matrix
            cov_det = np.linalg.det(sigma)  # determinant of covariance matrix

            x = np.linspace(
                np.min(x_train[:, 0]) - 2, np.max(x_train[:, 0]) + 2)
            y = np.linspace(
                np.min(x_train[:, 1]) - 2, np.max(x_train[:, 1]) + 2)
            X, Y = np.meshgrid(x, y)
            coe = 1.0 / ((2 * np.pi)**2 * cov_det)**0.5
            Z = coe * np.e ** (-0.5 * (cov_inv[0, 0]*(X-mu[0])**2 + (
                cov_inv[0, 1] + cov_inv[1, 0])*(X-mu[0])*(Y-mu[1]) +
                cov_inv[1, 1]*(Y-mu[1])**2))
            plt.contour(X, Y, Z)

        plot_contour(mu1, sigma1)
        plot_contour(mu2, sigma2)

        #plt.show()
        plt.savefig('Resultat2.png')
