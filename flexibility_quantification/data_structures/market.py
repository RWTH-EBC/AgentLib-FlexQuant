import pydantic


class MarketOptions(pydantic.BaseModel):
    name: str


class RandomOptions(MarketOptions):
    name: str = "random"
    random_seed: int = pydantic.Field(
        name="random_seed", default=None,
        description="Random seed for reproducing experiments"
    )

    pos_neg_rate: float = pydantic.Field(
        name="pos_neg_rate", default=0,
        description="Determines the likelihood positive and the negative flexibility."
                    "A higher rate means that more positive offers will be accepted.",
        ub=1, lb=0
    )

    offer_acceptance_rate: float = pydantic.Field(
        name="offer_acceptance_rate", default=0.5,
        description="Determines the likelihood of an accepted offer",
        ub=1, lb=0
    )


class SingleOptions(MarketOptions):
    name: str = "single"


class CustomOptions(MarketOptions):
    name: str = "custom"

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
        description="minimum average of an accepted offer"
    )

    options: MarketOptions = pydantic.Field(...,
                                            description="Market options, changes depending on 'type'")

    # Root validator to dynamically adjust 'options' based on 'type'
    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_and_set_options(cls, values):
        market_type = values.get('type')
        options_data = values.get('options')

        if market_type == 'random':
            values['options'] = RandomOptions(**options_data)
        elif market_type == 'custom':
            values['options'] = CustomOptions(**options_data)
        elif market_type == 'single':
            values['options'] = SingleOptions(**options_data)
        else:
            raise ValueError(f"Unknown market type: {market_type}")

        return values


class RandomMarket(MarketSpecifications):
    type: str = "random"


