import importlib

from models.arch import *


def make_model(name: str):
    base = importlib.import_module('models.' + name)
    if name.startswith('twostage'):
        model = getattr(base, 'TwoStageYTMTNetModel')
    else:
        model = getattr(base, 'YTMTNetModel')
    return model
