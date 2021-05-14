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

# def create_solver_v2(opt):
#     if opt['mode'] == 'sr':
#         solver = SRSolver2_v2(opt)
#     else:
#         raise NotImplementedError

#     return solver

# def create_solver_v3(opt):
#     if opt['mode'] == 'sr':
#         solver = SRSolver2_v3(opt)
#     else:
#         raise NotImplementedError

    return solver