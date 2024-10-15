from agentlib_mpc.modules import mpc_full
from flexibility_quantification.utils.data_handling import strip_multi_index
from typing import Dict, Union
from agentlib.core.datamodels import AgentVariable


class FlexibilityShadowMPC(mpc_full.MPC):
    # TODO: remove string handling
    config: mpc_full.MPCConfig

    def __init__(self, *args, **kwargs):
        # create instance variable
        self._full_controls: Dict[str, Union[AgentVariable, None]] = {}
        super().__init__(*args, **kwargs)

    def register_callbacks(self):
        for control_var in self.config.controls:
            self.agent.data_broker.register_callback(
                name=f"{control_var.name}_full", alias=f"{control_var.name}_full",
                callback=self.calc_flex_callback
            )
        for input_var in self.config.inputs:
            if input_var.name.replace("_", "", 1) in [control_var.name for control_var in self.config.controls]:
                self._full_controls[input_var.name.replace("_", "", 1)] = input_var

        super().register_callbacks()

    def calc_flex_callback(self, inp, name):
        """set the control trajectories before calculating the flexibility offer.
        self.model should account for flexibility in its cost function

        """
        # during provision dont calculate flex
        if self.get("in_provision").value:
            return

        vals = strip_multi_index(inp.value)

        # the MPC Predictions starts at t=env.now not t=0
        vals.index += self.env.time
        self._full_controls[name.replace("_full", "")].value = vals
        self.set(f"_{name.replace('_full', '')}", vals)
        # make sure all controls are set
        if all(x.value is not None for x in self._full_controls.values()):
            self.do_step()
            for name in self._full_controls.keys():
                self._full_controls[name.replace("_full", "")].value = None

    def process(self):
        # the shadow mpc should only be run after the results of the baseline are sent
        yield self.env.event()
