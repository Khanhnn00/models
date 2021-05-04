from .SRSolver import SRSolver
from .SRSolver2 import SRSolver2

def create_solver(opt):
    if opt['mode'] == 'sr':
        solver = SRSolver(opt)
    else:
        raise NotImplementedError

    return solver

def create_solver_split(opt):
    if opt['mode'] == 'sr':
        solver = SRSolver2(opt)
    else:
        raise NotImplementedError

    return solver