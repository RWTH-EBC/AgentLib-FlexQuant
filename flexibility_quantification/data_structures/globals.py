"""Script containing global variables"""

from typing import Literal


PREP_TIME = "prep_time"
MARKET_TIME = "market_time"
FLEX_EVENT_DURATION = "flex_event_duration"
POFILE_DEVIATION_WEIGHT = "profile_deviation_weight"
TIME_STEP = "time_step"
PREDICTION_HORIZON = "prediction_horizon"
FlexibilityOffer = "FlexibilityOffer"

FlexibilityDirections = Literal["positive", "negative"]

POWER_ALIAS_BASE = "__P_el_base"
POWER_ALIAS_NEG = "__P_el_neg"
POWER_ALIAS_POS = "__P_el_pos"

SHADOW_MPC_COST_FUNCTION = ("return ca.if_else(self.Time.sym < self.prep_time.sym + "
                            "self.market_time.sym, obj_std, ca.if_else(self.Time.sym < "
                            "(self.prep_time.sym + self.flex_event_duration.sym + "
                            "self.market_time.sym), obj_flex, obj_std))")

full_trajectory_suffix: str = "_full"
full_trajectory_prefix: str = "_"

def return_baseline_cost_function(power_variable):
    cost_func = ("return ca.if_else(self.in_provision.sym, "
                 "ca.if_else(self.Time.sym < self.rel_start.sym, obj_std, "
                 "ca.if_else(self.Time.sym >= self.rel_end.sym, obj_std, "
                 f"sum([self.profile_deviation_weight*(self.{power_variable} - "
                 "self._P_external)**2]))),obj_std)")
    return cost_func
