from pydantic import BaseModel
import pydantic
from agentlib.core.datamodels import _TYPE_MAP
import pandas as pd
from enum import Enum
from typing import Optional


class OfferStatus(Enum):
    not_accepted = "Not Accepted"
    accepted_positive = "Accepted Positive"
    accepted_negative = "Accepted Negative"


class FlexEnvelope(BaseModel):
    """
    Data class for the flexibility envelope
    """
    energyflex_pos: pd.Series = pydantic.Field(
        default=None,
        scalar=False,
        description="",
    )
    energyflex_neg: pd.Series = pydantic.Field(
        default=None,
        scalar=False,
        description="",
    )
    energyflex_base: pd.Series = pydantic.Field(
        default=None,
        scalar=False,
        description="",
    )
    time_steps: list = pydantic.Field(
        default=None,
        scalar=False,
        description="",
    )
    powerflex_pos: pd.Series = pydantic.Field(
        default=None,
        scalar=False,
        description="",
    )
    powerflex_neg: pd.Series = pydantic.Field(
        default=None,
        scalar=False,
        description="",
    )
    powerflex_base: pd.Series = pydantic.Field(
        default=None,
        scalar=False,
        description="",
    )
    p_el_max: float = pydantic.Field(
        default=None,
        scalar=True,
        description="",
    )
    p_el_min: float = pydantic.Field(
        default=None,
        scalar=True,
        description="",
    )

    class Config:
        arbitrary_types_allowed = True

    def as_dataframe(self) -> pd.DataFrame:
        flex_env_data = {'energyflex_pos': self.energyflex_pos.to_list(),
                         'energyflex_neg': self.energyflex_neg.tolist(),
                         'energyflex_base': self.energyflex_base.tolist(),
                         'time_steps': self.time_steps,
                         'powerflex_pos': self.powerflex_pos.to_list(),
                         'powerflex_neg': self.powerflex_neg.to_list(),
                         'powerflex_base': self.powerflex_base.to_list(),
                         'p_el_max': self.p_el_max,
                         'p_el_min': self.p_el_min,
                         }

        return pd.DataFrame(list(flex_env_data.items()), columns=['Keys', 'Values'], index=list(flex_env_data.keys()))


class FlexOffer(BaseModel):
    """Data class for the flexibility offer

    """
    base_power_profile: pd.Series = pydantic.Field(
        default=None,
        unit="W",
        scalar=False,
        description="Power profile of the baseline MPC",
    )
    pos_price: float = pydantic.Field(
        default=None,
        unit="ct",
        scalar=True,
        description="Price for positive flexibility",
    )
    pos_diff_profile: pd.Series = pydantic.Field(
        default=None,
        unit="W",
        scalar=False,
        description="Power profile for the positive difference",
    )
    neg_price: float = pydantic.Field(
        default=None,
        unit="ct",
        scalar=True,
        description="Price for negative flexibility",
    )
    neg_diff_profile: pd.Series = pydantic.Field(
        default=None,
        unit="W",
        scalar=False,
        description="Power profile for the negative difference",
    )
    status: OfferStatus = pydantic.Field(
        default=OfferStatus.not_accepted,
        scalar=True,
        description="Status of the FlexOffer",
    )
    flex_envelope: Optional[FlexEnvelope] = pydantic.Field(
        default=None,
        scalar=False,
        description="Flexibility envelope of the FlexOffer",
    )

    class Config:
        arbitrary_types_allowed = True

    def as_dataframe(self):
        """Returns the offer as a dataframe. Scalar values are written on the first timestep

        """
        data = []
        cols = []

        # append scalar values
        for name, field in self.model_fields.items():
            if field.json_schema_extra["scalar"]:
                ser = pd.Series(getattr(self, name))
                ser.index += self.base_power_profile.index[0]
                data.append(ser)
                cols.append(name)

        df = pd.DataFrame(data).T
        df.columns = cols
        return df


# add the offer type to agent variables
_TYPE_MAP["FlexOffer"] = FlexOffer
