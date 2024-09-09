from pydantic import BaseModel
from typing import TypeVar
from agentlib.core.datamodels import _TYPE_MAP
import pandas as pd
from io import StringIO
import json
from enum import Enum
from pandas.core.frame import DataFrame

class OfferStatus(Enum):
    not_accepted = "Not Accepted"
    accepted_positive = "Accepted Positive"
    accepted_negative = "Accepted Negative"


DataFrame = TypeVar('DataFrame')

class PowerFlexOffer(BaseModel):
    """
    The Pydantic Variable for the power flexibility offer
    """
    base_power_profile: DataFrame
    pos_price: float
    pos_diff_profile: DataFrame
    pos_time_flex: float
    comfort_violation_pos: float = 0
    neg_price: float
    neg_diff_profile: DataFrame
    neg_time_flex: float
    comfort_violation_neg: float = 0
    power_multiplier: float = 1

    status: OfferStatus = OfferStatus.not_accepted

    def dataframe(self):
        """
        Returns the offer as a dataframe. Scalar values are written on the first timestep
        """
        results = []
        results.append(self.base_power_profile)
        results.append(self.pos_diff_profile)
        results.append(self.neg_diff_profile)
        for field in (self.pos_price, self.pos_time_flex, self.comfort_violation_pos, 
                      self.neg_price, self.neg_time_flex, self.comfort_violation_neg, 
                      self.power_multiplier,self.status.value):
            ser = pd.Series(field)
            ser.index += self.base_power_profile.index[0]
            results.append(ser)
        
        df = pd.DataFrame(results).T
        df.columns = ["Base Profile", "Positive Profile", "Negative Profile", 
                      "Positive Price", "Positive Timeflex", "Positive Comfort Violation", 
                      "Negative Price", "Negative Timeflex", "Negative Comfort Violation", 
                      "Power Multiplier", "Status"]
        return df
        



    
# add the offer type to agent variables
_TYPE_MAP["PowerFlexOffer"] = PowerFlexOffer
