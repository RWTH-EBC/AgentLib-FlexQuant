import pydantic
import pandas as pd
from enum import Enum
from pydantic import BaseModel
from typing import Optional
from agentlib.core.datamodels import _TYPE_MAP


class OfferStatus(Enum):
    not_accepted = "Not Accepted"
    accepted_positive = "Accepted Positive"
    accepted_negative = "Accepted Negative"


class FlexOffer(BaseModel):
    """Data class for the flexibility offer."""
    base_power_profile: pd.Series = pydantic.Field(
        default=None,
        unit="W",
        scalar=False,
        description="Power profile of the baseline MPC",
    )
    pos_price: Optional[float] = pydantic.Field(
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
    neg_price: Optional[float] = pydantic.Field(
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
        default=OfferStatus.not_accepted.value,
        scalar=True,
        description="Status of the FlexOffer",
    )

    class Config:
        arbitrary_types_allowed = True

    def as_dataframe(self) -> pd.DataFrame:
        """Store the flexibility offer in a pd.DataFrame

        Returns:
            DataFrame containing the flexibility offer.
            Scalar values are written on the first timestep.

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
