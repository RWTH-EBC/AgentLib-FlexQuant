from agentlib_mpc.modules import mpc_full
from flexibility_quantification.utils.data_handling import strip_multi_index


class FlexibilityShadowMPC(mpc_full.MPC):
    config: mpc_full.MPCConfig

    def register_callbacks(self):
        self.__controls = {}
        for control in self.var_ref.controls:
            self.agent.data_broker.register_callback(
                name=f"{control}_full", alias=f"{control}_full", callback=self.calc_flex_callback
            )
            self.__controls[control] = None
        super().register_callbacks()

    def calc_flex_callback(self, inp, name):
        """
        set the control trajectories before calculating the flexibility offer.
        self.model should account for flexibility in its cost function
        """
        # during provision dont calculate flex  TODO: calculate after ending of event
        if self.get("in_provision").value:
            return

        vals = strip_multi_index(inp.value)

        # The MPC Predictions starts at t=env.now not t=0!
        vals.index += self.env.time
        self.__controls[name.replace("_full", "")] = vals
        self.set(f"_{name.replace('_full', '')}", vals)
        # make sure all controls are set
        if all(x is not None for x in self.__controls.values()):
            self.do_step()
            for x in self.__controls.keys():
                self.__controls[x.replace("_full", "")] = None

    def process(self):
        # the shadow mpc should only be run after the results of the baseline are sent
        yield self.env.event()
