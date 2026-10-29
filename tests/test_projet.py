"""Tests rapides.   Usage : python -m pytest  (depuis la racine du projet)"""
import numpy as np
import pandas as pd
import torch

from pcg.donnees import BASE, lire, plis
from pcg.mfcc import DUREE, FS, decouper, dossier, mfcc
from pcg.modele import TCResNet8, compter
from pcg.train import macc, metriques, normaliser_enregistrements, probas

Y = np.array([1, 1, -1, -1, -1, -1])


def test_macc():
    assert macc(Y, np.full(6, -1))[0] == 0.5                                # tout "normal" -> 0,5
    assert macc(Y, Y)[0] == 1.0                                             # prédiction parfaite
    assert macc(Y, np.array([1, -1, -1, -1, -1, 1])) == (0.625, 0.5, 0.75)  # calculé à la main


def test_metriques():
    m = metriques(Y, np.array([0.9, 0.2, 0.1, 0.4, 0.6, 0.3]))  # 1 anormal sur 2 trouvé, 3 normaux sur 4
    assert (m["macc"], m["se"], m["sp"]) == (0.625, 0.5, 0.75)
    assert m["auc"] == 0.625                                    # 5 paires anormal/normal sur 8 bien ordonnées


def test_mfcc():
    fenetres = decouper(np.random.randn(10000))
    assert fenetres.shape[1] == int(DUREE * FS)
    assert mfcc(fenetres[0]).shape == (13, 75)


def test_fin_du_signal():
    x = np.arange(10000.0)
    f = decouper(x)
    assert f[-1][-1] == x[-1]                                   # la fin du signal n'est plus perdue
    assert decouper(np.arange(1000.0)).shape == (1, int(DUREE * FS))  # signal trop court : une fenêtre, sans erreur


def test_modele():
    assert compter(TCResNet8()) == (25056, 429630)              # paramètres, MACs
    assert TCResNet8().eval()(torch.zeros(4, 13, 75)).shape == (4,)


def test_reseaux_de_comparaison():
    from pcg.modele import RESEAUX
    for nom, reseau in RESEAUX.items():
        assert reseau().eval()(torch.zeros(4, 13, 75)).shape == (4,), nom       # même entrée, même sortie pour tous
        assert compter(reseau())[1] > 0, nom


def test_variantes_d_entrainement(monkeypatch):
    """Les options aug et adv tournent et la tête adversaire ne change pas le modèle sauvegardé ; l'inversion de gradient change le signe."""
    from pcg import train
    monkeypatch.setattr(train, "EPOQUES", 1)
    X = torch.randn(40, 13, 75, device=train.DEV)
    y = torch.tensor([0.0, 1.0] * 20, device=train.DEV)
    w = torch.ones(40, device=train.DEV)
    b = torch.tensor([0, 1, 2, 3] * 10, device=train.DEV)
    label = pd.Series([-1, 1], index=["r0", "r1"])
    ids = np.array(["r0", "r1"] * 10)
    for aug, adv in [(False, 0.0), (True, 0.0), (False, 0.3), (True, 0.3)]:
        modele, _ = train.ajuster(X, y, w, X[:20], ids, label, "tcresnet8", aug, adv, b)
        assert compter(modele) == (25056, 429630)
    x = torch.ones(3, requires_grad=True)
    (train.Inversion.apply(x, 0.5) * 2).sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -1.0))


def test_plis():
    df = lire()
    for nom, p in plis(df).items():
        # aucun enregistrement dans deux parties, et tous sont utilisés
        assert not (p["train"] & p["val"]) and not (p["train"] & p["test"]) and not (p["val"] & p["test"]), nom
        assert len(p["train"] | p["val"] | p["test"]) == len(df), nom
        if nom.startswith("lodo"):
            base = lambda ids: set(df[df.id.isin(ids)].base)
            assert base(p["test"]) == {nom[-1]}                                  # le test est la base entière
            assert len(base(p["val"])) == 1 and base(p["val"]) != {nom[-1]}      # la validation, une autre base entière
            assert not base(p["train"]) & (base(p["val"]) | {nom[-1]})           # ni l'une ni l'autre dans le train
    assert not plis(df)["tout"]["test"] and set(df[df.id.isin(plis(df)["tout"]["val"])].base) == {"a"}


def test_normalisation_par_enregistrement():
    feats = (np.random.randn(6, 13, 75) * 3 + 5).astype(np.float32)
    ids = np.array(["a", "a", "a", "b", "b", "b"])
    sortie = normaliser_enregistrements(feats, ids)
    for idx in (slice(0, 3), slice(3, 6)):                      # chaque enregistrement : moyenne 0 et écart-type 1
        assert np.allclose(sortie[idx].mean(axis=(0, 2)), 0, atol=1e-4)
        assert np.allclose(sortie[idx].std(axis=(0, 2)), 1, atol=1e-3)


def test_sauvegarde_et_prediction(tmp_path, monkeypatch):
    """Les poids se rechargent à l'identique, et pcg.predire (à partir du .wav) redonne le calcul de l'entraînement."""
    from pcg import predire, train
    monkeypatch.setattr(train, "MODELES", tmp_path / "models")
    monkeypatch.setattr(train, "PREDICTIONS", tmp_path / "pred")
    torch.manual_seed(0)
    modele = TCResNet8()
    res = dict(pli="tout", graine=0)
    train.sauvegarder(res, modele, np.zeros((1, 13, 1), np.float32), np.ones((1, 13, 1), np.float32), None, None)
    rec, mu, sd, res2 = train.charger(tmp_path / "models" / "meilleur_modele.pt")
    x = torch.randn(3, 13, 75)
    assert torch.equal(modele.eval()(x), rec(x)) and res2 == res

    # enregistrement a0001 : prédiction depuis le .wav = moyenne des fenêtres des MFCC déjà calculés et normalisés
    feats, fen = np.load(dossier() / "features.npy"), pd.read_csv(dossier() / "fenetres.csv")
    m = (fen.id == "a0001").to_numpy()
    X = normaliser_enregistrements(feats[m], fen.id[m].to_numpy())
    attendu = probas(rec, torch.from_numpy(X).float().to(train.DEV)).mean()
    wav = next(BASE.glob("training-a")) / "a0001.wav"
    assert np.isclose(predire.predire(wav, tmp_path / "models" / "meilleur_modele.pt"), attendu, atol=1e-5)
