import pydantic
from typing import Union, Literal
from flexibility_quantification.data_structures.globals import FlexibilityDirections


class RandomOptions(pydantic.BaseModel):
    type: Literal["random"]
    random_seed: int = pydantic.Field(
        name="random_seed", default=None,
        description="Random seed for reproducing experiments"
    )

    pos_neg_rate: float = pydantic.Field(
        name="pos_neg_rate", default=0,
        description="Determines the likelihood positive and the negative flexibility."
                    "A higher rate means that more positive offers will be accepted.",
        le=1, ge=0
    )

    offer_acceptance_rate: float = pydantic.Field(
        name="offer_acceptance_rate", default=0.5,
        description="Determines the likelihood of an accepted offer",
        le=1, ge=0
    )


class SingleOptions(pydantic.BaseModel):
    type: Literal["single"]
    start_time: float = pydantic.Field(description="After this time, the first available flex offer"
                                                   " is accepted")
    direction: FlexibilityDirections = pydantic.Field(default="positive", description="Direction of the flexibility")


class CustomOptions(pydantic.BaseModel):
    type: Literal["custom"]

    model_config = pydantic.ConfigDict(extra="allow")


class MarketSpecifications(pydantic.BaseModel):
    type: str = pydantic.Field(
        default=None,
        description="Name of market type"
    )

    cooldown: int = pydantic.Field(
        name="cooldown", default=6,
        description="cooldown time (no timesteps) after a provision"
    )

    minimum_average_flex: float = pydantic.Field(
        name="minimum_average_flex", default=0,
        unit="W",
        description="minimum average of an accepted offer"
    )

    options: Union[RandomOptions, SingleOptions, CustomOptions] = pydantic.Field(
        ...,
        description="Market options, changes depending on 'type'",
        discriminator='type'
    )

    # Root validator to automatically populate the options.type from the top-level type
    @pydantic.model_validator(mode='before')
    @classmethod
    def set_options_type(cls, values):
        market_type = values.get('type')
        options = values.get('options', {})

        # Ensure the options dict contains the correct 'type' field
        if isinstance(options, dict) and 'type' not in options:
            options['type'] = market_type
            values['options'] = options

        return values


class RandomMarket(MarketSpecifications):
    type: str = "random"
