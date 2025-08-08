# AceCG/trainers/analytic.py
import numpy as np
from .base import BaseREMTrainer
from ..utils.compute import dUdLByFrame, dUdL, d2UdLjdLk_Matrix, dUdLj_dUdLk_Matrix, Hessian

class REMTrainerAnalytic(BaseREMTrainer):
    def step(self, AA_data, CG_data, step_index: int = 0):
        # === Compute dS/dL ===
        dUdL_AA_frame = dUdLByFrame(self.potential, AA_data['dist'])
        dUdL_CG_frame = dUdLByFrame(self.potential, CG_data['dist'])
        dUdL_AA = dUdL(dUdL_AA_frame, AA_data.get('weight'))
        dUdL_CG = dUdL(dUdL_CG_frame, CG_data.get('weight'))
        dSdL = self.beta * (dUdL_AA - dUdL_CG)

        # === Compute Hessian (for NR only) ===
        H = None
        if hasattr(self.optimizer, 'step') and 'hessian' in self.optimizer.step.__code__.co_varnames:
            d2U = d2UdLjdLk_Matrix(self.potential, CG_data['dist'], CG_data.get('weight'))
            dUU = dUdLj_dUdLk_Matrix(dUdL_CG_frame, CG_data.get('weight'))
            H = Hessian(self.beta, d2U, dUU, dUdL_CG)

        # === Optimization Step ===
        update = self.optimizer.step(dSdL, hessian=H)

        # === Logging ===
        if self.logger:
            self.logger.add_scalar("REM/grad_norm", np.linalg.norm(dSdL), step_index)
            self.logger.add_scalar("REM/update_norm", np.linalg.norm(update), step_index)
            if H is not None:
                self.logger.add_scalar("REM/hessian_cond", np.linalg.cond(H), step_index)

        return dSdL, update