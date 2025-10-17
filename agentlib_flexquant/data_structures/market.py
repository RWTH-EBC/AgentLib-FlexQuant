"""
Market specification data models for flexibility offer acceptance strategies.

Defines Pydantic models for configuring different market behaviors including
random acceptance, single offer acceptance, and custom strategies with
associated parameters and validation.
"""
from typing import Literal, Union

import pydantic

from agentlib_flexquant.data_structures.globals import FlexibilityDirections, COLLOCATION, CONSTANT


class RandomOptions(pydantic.BaseModel):
    """Configuration options for random market behavior."""

    type: Literal["random"]
    random_seed: int = pydantic.Field(
        name="random_seed",
        default=None,
        description="Random seed for reproducing experiments",
    )

    pos_neg_rate: float = pydantic.Field(
        name="pos_neg_rate",
        default=0,
        description="Determines the likelihood positive and the negative flexibility."
        "A higher rate means that more positive offers will be accepted.",
        le=1,
        ge=0,
    )

    offer_acceptance_rate: float = pydantic.Field(
        name="offer_acceptance_rate",
        default=0.5,
        description="Determines the likelihood of an accepted offer",
        le=1,
        ge=0,
    )


class SingleOptions(pydantic.BaseModel):
    """Configuration options for single offer acceptance market behavior."""

    type: Literal["single"]
    offer_acceptance_time: float = pydantic.Field(
        description="After this time, the first available flex offer is accepted"
    )
    direction: FlexibilityDirections = pydantic.Field(
        default="positive", description="Direction of the flexibility"
    )


class CustomOptions(pydantic.BaseModel):
    """Configuration options for custom market behavior with flexible parameters."""

    type: Literal["custom"]

    model_config = pydantic.ConfigDict(extra="allow")


class MarketSpecifications(pydantic.BaseModel):
    """Base specification for flexibility market behavior and parameters."""

    type: str = pydantic.Field(default=None, description="Name of market type")

    cooldown: int = pydantic.Field(
        name="cooldown",
        default=6,
        description="cooldown time (no timesteps) after a provision",
    )

    minimum_average_flex: float = pydantic.Field(
        name="minimum_average_flex",
        default=0,
        unit="W",
        description="minimum average of an accepted offer",
    )

    options: Union[RandomOptions, SingleOptions, CustomOptions] = pydantic.Field(
        ...,
        description="Market options, changes depending on 'type'",
        discriminator="type",
    )

    accepted_offer_sample_points: Literal[COLLOCATION, CONSTANT] = pydantic.Field(
        default='collocation',
        description="Method defining how to send the accepted flexibility power back"
                    "to the baseline mpc so that the system can deliver it. "
                    "This is relevant for collocation, as the power profile is only "
                    "defined at the collocation points. If you choose constant here, "
                    "the values are averaged and set for the whole time step."
                    "Set to constant if the power only depends on control or if "
                    "system has low inertial. Otherwise, use collocation.",
    )

    # Root validator to automatically populate the options.type from the top-level type
    @pydantic.model_validator(mode="before")
    @classmethod
    def set_options_type(cls, values):
        """Automatically set the options type field from the top-level market type."""
        market_type = values.get("type")
        options = values.get("options", {})

        # Ensure the options dict contains the correct 'type' field
        if isinstance(options, dict) and "type" not in options:
            options["type"] = market_type
            values["options"] = options

        return values


class RandomMarket(MarketSpecifications):
    """Market specification for random flexibility offer acceptance behavior."""

    type: str = "random"
