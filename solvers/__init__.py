from .SRSolver import SRSolver
from .SRSolver2 import SRSolver2
from .SRSolver_v2 import SRSolver_v2
from .SRSolver_v3 import SRSolver_v3
from .SRSolver3 import SRSolver3
from .SRSolver4 import SRSolver4

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

def create_solver_v2(opt):
    if opt['mode'] == 'sr':
        solver = SRSolver_v2(opt)
    else:
        raise NotImplementedError

    return solver

def create_solver_v3(opt):
    if opt['mode'] == 'sr':
        solver = SRSolver_v3(opt)
    else:
        raise NotImplementedError

    return solver

def create_solver_abla(opt):
    if opt['mode'] == 'sr':
        solver = SRSolver3(opt)
    else:
        raise NotImplementedError

    return solver

def create_solver_v4(opt):
    if opt['mode'] == 'sr':
        solver = SRSolver4(opt)
    else:
        raise NotImplementedError

    return solver