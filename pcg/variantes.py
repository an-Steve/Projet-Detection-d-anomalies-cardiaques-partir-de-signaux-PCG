"""Deux variantes d'entraînement testées pour améliorer le LODO, désactivées par défaut (voir README, « Essayer d'améliorer le LODO »).

Aucune n'a amélioré le résultat de référence : le modèle livré n'en utilise pas.
  - augmenter : perturbe les MFCC pendant l'entraînement (option --augmenter).
  - Inversion : inversion de gradient pour l'entraînement adversaire contre la base d'origine (option --adversaire).
"""
import torch


def augmenter(x):
    """Perturbe un lot de MFCC : bruit, décalage de moyenne (autre micro) et décalage dans le temps."""
    x = x + 0.1 * torch.randn_like(x) + 0.3 * torch.randn(x.shape[0], x.shape[1], 1, device=x.device)
    return torch.roll(x, int(torch.randint(-10, 11, (1,))), dims=2)


class Inversion(torch.autograd.Function):
    """Identité à l'aller ; au retour, le gradient est changé de signe et multiplié par lam."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None
