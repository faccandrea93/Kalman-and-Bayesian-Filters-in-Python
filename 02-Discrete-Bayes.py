# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 10:15:05 2021

@author: facca

Script per la costruzione di un filtro Bayesiano
Esempio riportato dal rispettivo Notebook, esempio del
tracciamento di un cane

"""
import numpy as np
import book_format
import kf_book.book_plots as book_plots
from kf_book.book_plots import figsize, set_figsize
import matplotlib.pyplot as plt
from filterpy.discrete_bayes import normalize, update

def update_belief_hallway(hallway, belief,  z, z_prob):
    # Likelihood = stima basata sui dati : p(x|theta)
    likelihood = np.ones(len(hallway))
    scale = z_prob / (1 - z_prob)
    # NumPy Booleans: dove z (misura) coincide con la posizione della porta
    likelihood[hallway==z]*=scale
    # =============================================================================
    # Likelihood non è una pdf (sum not 1)
    # =============================================================================
    # Bayes's Formula : to normalize
    # posterior = likelihoo*prior/normalize
    return normalize(likelihood*belief)



# 1 = doors ; 0 = walls
hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])

# Probabilità A PRIORI di dove possa trovarsi il cane, senza
# alcuna informazione -- Perfect Sensors
belief = np.array([1/3, 1/3, 0, 0, 0, 0, 0, 0, 1/3, 0])
book_plots.bar_plot(belief)

# Probabilità A PRIORI di dove possa trovarsi il cane, senza
# alcuna informazione -- Perfect Sensors
belief = np.array([.31, .31, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, .31, .01])
book_plots.bar_plot(belief)
# Probabilità a Priori: secondo l'approccio Bayesiano questa quantità
# viene poi aggiornata

# Calcolo della probabilità a posteriori
reading = 1 # viene misurato un 1 = door
# z_prob serve per definire quanto è più attendibile una misura su una "door"
# rispetto ad un altro punto
# belief: è l'ipotesi di dove si trovi il cane che facciamo a priori, data la misura che
# abbiamo rilevato
posterior = update_belief_hallway(hallway, belief, z=reading,z_prob=0.75)
book_plots.bar_plot(posterior)
print(f"sum: {sum(posterior)}")



