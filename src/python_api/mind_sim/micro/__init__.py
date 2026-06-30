from .. import _native


def sim():
    return _native.Sim()


__all__ = ["sim"]
