from agentlib_mpc.optimization_backends.casadi_.minlp_cia import CasADiCIABackend
from agentlib_mpc.optimization_backends.casadi_.core.casadi_backend import (
    CasadiBackendConfig,
)
from agentlib_mpc.data_structures.mpc_datamodels import DiscretizationOptions
import pydantic
from pydantic import ConfigDict
from typing import Optional
from pathlib import Path
from agentlib.core.errors import OptionalDependencyError
from agentlib_mpc.data_structures.mpc_datamodels import MINLPVariableReference
from agentlib_flexquant.data_structures.globals import full_trajectory_suffix

try:
    import pycombina
except ImportError:
    raise OptionalDependencyError(
        used_object="Pycombina",
        dependency_install=".\ after cloning pycombina. Instructions: "
        "https://pycombina.readthedocs.io/en/latest/install.html#",
    )


class ConstrainedCIABackendConfig(CasadiBackendConfig):
    market_time: int = pydantic.Field(
        default=900,
        ge=0,
        unit="s",
        description="Time for market interaction",
    )

    class Config:
        # Explicitly set this to allow additional fields in the derived class
        extra = "forbid"


class ConstrainedCasADiCIABackend(CasADiCIABackend):
    var_ref: MINLPVariableReference
    config_type = ConstrainedCIABackendConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_pycombina(self, b_rel):
        # N = self.discretization.options.prediction_horizon
        # dt = self.discretization.options.time_step
        # time_end = N * dt
        grid = self.discretization.grid(self.system.binary_controls).copy()
        grid.append(grid[-1] + self.config.discretization_options.time_step)
        # grid = np.linspace(0, time_end, N + 1)

        binapprox = pycombina.BinApprox(
            t=grid,
            b_rel=b_rel,
        )

        # constrain shadow MPCs to values of baseline for time<market_time
        for bin_con in self.var_ref.binary_controls:
            # check for baseline or shadow MPC
            if (
                bin_con + full_trajectory_suffix
                not in self.model.get_input_names()
            ):
                continue
            # if shadow MPC, get current value send by baseline and constrain pycombina
            elif (
                    self.model.get_input(
                        bin_con + full_trajectory_suffix
                    ).value
                    is not None
                ):
                    cons = self.model.get_input(
                        bin_con + full_trajectory_suffix
                    ).value
                    cons = cons[cons.index < self.config.market_time]
                    last_idx = 0
                    for idx, value in cons.items():
                        # constrain ever timestep before market_time with values of baseline
                        binapprox.set_valid_controls_for_interval(
                            (last_idx, idx), [value, 1 - value]
                        )
                        last_idx = idx

        # binapprox.set_n_max_switches([3, 1, 0])
        # binapprox.set_min_up_times([3600, 1800, 0])

        bnb = pycombina.CombinaBnB(binapprox)
        bnb.solve(
            use_warm_start=False,
            max_cpu_time=15,
            verbosity=0,
        )
        b_bin = binapprox.b_bin

        # if there is only one mode, we created a dummy mode which we remove now
        if len(self.var_ref.binary_controls) == 1:
            b_bin = b_bin[0, :].reshape(1, -1)

        return b_bin
