from agentlib_flexquant import flexibility_market
from agentlib.core.datamodels import AgentVariable
from pydantic import Field

class CustomMarketConfig(flexibility_market.FlexibilityMarketModuleConfig):
    custom_string: str = Field(
        title="Custom String",
        description="Just an example of a custom String.",
    )

class CustomMarketModule(flexibility_market.FlexibilityMarketModule):
    config: CustomMarketConfig

    def custom_flexibility_callback(self, inp: AgentVariable, name: str):
        """Placeholder for a custom flexibility callback."""
        offer = inp.value
        offer.status = "Custom status"
        self.write_results(offer=offer)

        print("My custom string is: " + self.config.custom_string)
        print("Recieved offer: " + str(offer))
        pass