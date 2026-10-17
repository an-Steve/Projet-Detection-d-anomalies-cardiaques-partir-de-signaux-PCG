"""Tests rapides.   Usage : python -m pytest"""
import numpy as np
import torch
from donnees import lire, plis
from mfcc import FENETRE, decouper, mfcc
from modele import TCResNet8, compter
from train import macc

Y = np.array([1, 1, -1, -1, -1, -1])


def test_macc():
    assert macc(Y, np.full(6, -1))[0] == 0.5                                # tout "normal" -> 0,5
    assert macc(Y, Y)[0] == 1.0                                             # prédiction parfaite
    assert macc(Y, np.array([1, -1, -1, -1, -1, 1])) == (0.625, 0.5, 0.75)  # calculé à la main


def test_mfcc():
    fenetres = decouper(np.random.randn(10000))
    assert fenetres.shape[1] == FENETRE
    assert mfcc(fenetres[0]).shape == (13, 75)


def test_modele():
    assert compter(TCResNet8()) == (16169, 279648)  # paramètres, MACs
    assert TCResNet8().eval()(torch.zeros(4, 13, 75)).shape == (4,)


def test_plis():
    df = lire()
    for nom, p in plis(df).items():
        # aucun enregistrement dans deux parties, et tous sont utilisés
        assert not (p["train"] & p["val"]) and not (p["train"] & p["test"]) and not (p["val"] & p["test"]), nom
        assert len(p["train"] | p["val"] | p["test"]) == len(df), nom
        if nom.startswith("lodo"):  # la base de test n'apparaît jamais dans le train
            assert not df[df.id.isin(p["train"] | p["val"])].base.eq(nom[-1]).any()
