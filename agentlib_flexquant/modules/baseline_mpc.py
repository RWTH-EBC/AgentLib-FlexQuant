"""
Defines MPC and MINLP-MPC for baseline flexibility quantification.
"""
from agentlib_mpc.modules import minlp_mpc, mpc_full


class FlexibilityBaselineMPC(mpc_full.MPC):
    """MPC for baseline flexibility quantification."""

    config: mpc_full.MPCConfig

    def pre_computation_hook(self):
        """Calculate relative start and end times for flexibility provision.

        When in provision mode, computes the relative timing for flexibility
        events based on the external power profile timestamps and current
        environment time.
        """
        if self.get("in_provision").value:
            timestep = (
                self.get("_P_external").value.index[1]
                - self.get("_P_external").value.index[0]
            )
            self.set(
                "rel_start", self.get("_P_external").value.index[0] - self.env.time
            )
            # the provision profile gives a value for the start of a time step.
            # For the end of the flex interval add time step!
            self.set(
                "rel_end",
                self.get("_P_external").value.index[-1] - self.env.time + timestep,
            )


class FlexibilityBaselineMINLPMPC(minlp_mpc.MINLPMPC):
    """MINLP-MPC for baseline flexibility quantification with mixed-integer optimization."""

    config: minlp_mpc.MINLPMPCConfig

    def pre_computation_hook(self):
        """Calculate relative start and end times for flexibility provision.

        When in provision mode, computes the relative timing for flexibility
        events based on the external power profile timestamps and current
        environment time.
        """
        if self.get("in_provision").value:
            timestep = (
                self.get("_P_external").value.index[1]
                - self.get("_P_external").value.index[0]
            )
            self.set(
                "rel_start", self.get("_P_external").value.index[0] - self.env.time
            )
            # the provision profile gives a value for the start of a time step.
            # For the end of the flex interval add time step!
            self.set(
                "rel_end",
                self.get("_P_external").value.index[-1] - self.env.time + timestep,
            )
