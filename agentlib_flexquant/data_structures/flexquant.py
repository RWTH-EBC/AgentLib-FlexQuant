import pydantic
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union
from pydantic import ConfigDict, model_validator
from agentlib.core.agent import AgentConfig
from agentlib.core.errors import ConfigurationError
from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable
from agentlib_flexquant.data_structures.mpcs import BaselineMPCData, NFMPCData, PFMPCData


class ForcedOffers(Enum):
    positive = "positive"
    negative = "negative"


class ShadowMPCConfigGeneratorConfig(pydantic.BaseModel):
    """Class defining the options to initialize the shadow mpc config generation."""
    model_config = ConfigDict(
        json_encoders={MPCVariable: lambda v: v.dict()},
        extra='forbid'
    )    
    weights: List[MPCVariable] = pydantic.Field(
        default=[],
        description="Name and value of weights",
    )
    pos_flex: PFMPCData = pydantic.Field(
        default=None,
        description="Data for PF-MPC"
    )
    neg_flex: NFMPCData = pydantic.Field(
        default=None,
        description="Data for NF-MPC"
    )
    @model_validator(mode="after")
    def assign_weights_to_flex(self):
        if self.pos_flex is None:
            raise ValueError("Missing required field: 'pos_flex' specifying the pos flex cost function.")
        if self.neg_flex is None:
            raise ValueError("Missing required field: 'neg_flex' specifying the neg flex cost function.")
        if self.weights:
            self.pos_flex.weights = self.weights
            self.neg_flex.weights = self.weights
        return self


class FlexibilityMarketConfig(pydantic.BaseModel):
    """Class defining the options to initialize the market."""
    model_config = ConfigDict(
        extra='forbid'
    )
    agent_config: AgentConfig
    name_of_created_file: str = pydantic.Field(
        default="flexibility_market.json",
        description="Name of the config that is created by the generator",
    )


class FlexibilityIndicatorConfig(pydantic.BaseModel):
    """Class defining the options for the flexibility indicators."""
    model_config = ConfigDict(
        json_encoders={Path: str, AgentConfig: lambda v: v.model_dump()},
        extra='forbid'
    )
    agent_config: AgentConfig
    name_of_created_file: str = pydantic.Field(
        default="indicator.json",
        description="Name of the config that is created by the generator",
    )
    @model_validator(mode="after")
    def check_file_extension(self):
        if self.name_of_created_file: 
            file_path = Path(self.name_of_created_file)
            if file_path.suffix != ".json":
                raise ConfigurationError(
                    f"Invalid file extension for name_of_created_file: '{self.name_of_created_file}'. "
                    f"Expected a '.json' file."
                )
        return self


class FlexQuantConfig(pydantic.BaseModel):
    """Class defining the options to initialize the FlexQuant generation."""
    model_config = ConfigDict(
        json_encoders={Path: str},
        extra='forbid'
    )
    prep_time: int = pydantic.Field(
        default=1800,
        ge=0,
        unit="s",
        description="Preparation time before the flexibility event",
    )
    flex_event_duration: int = pydantic.Field(
        default=7200,
        ge=0,
        unit="s",
        description="Flexibility event duration",
    )
    market_time: int = pydantic.Field(
        default=900,
        ge=0,
        unit="s",
        description="Time for market interaction",
    )
    indicator_config: Union[FlexibilityIndicatorConfig, Path] = pydantic.Field(
        description="Path to the file or dict of flexibility indicator config",
    )
    market_config: Optional[Union[FlexibilityMarketConfig, Path]] = pydantic.Field(
        default=None,
        description="Path to the file or dict of market config",
    )
    baseline_config_generator_data: BaselineMPCData = pydantic.Field(
        description="Baseline generator data config file or dict",
    )
    shadow_mpc_config_generator_data: ShadowMPCConfigGeneratorConfig = pydantic.Field(
        description="Shadow mpc generator data config file or dict",
    )
    flex_base_directory_path: Optional[Path] = pydantic.Field(
        default_factory=lambda: Path.cwd() / "flex_output_data",
        description="Base path where generated flex data is stored",
    )
    flex_files_directory: Path = pydantic.Field(
        default=Path("created_flex_files"),
        description="Directory where generated files (jsons) should be stored",
    )
    results_directory: Path = pydantic.Field(
        default=Path("results"),
        description="Directory where generated result files (CSVs) should be stored",
    )
    delete_files: bool = pydantic.Field(
        default=True,
        description="If generated files should be deleted afterwards",
    )
    overwrite_files: bool = pydantic.Field(
        default=False,
        description="If generated files should be overwritten by new files",
    )

    @model_validator(mode="after")
    def check_config_file_extension(self):
        """
        Validates that the indicator and market config file paths have a '.json' extension.
    
        Raises:
            ValueError: If either file does not have the expected '.json' extension.
        """
        if isinstance(self.indicator_config, Path) and self.indicator_config.suffix != ".json":
            raise ValueError(
                f"Invalid file extension for indicator config: '{self.indicator_config}'. "
                f"Expected a '.json' file."
            )
        if isinstance(self.market_config, Path) and self.market_config.suffix != ".json":
            raise ValueError(
                f"Invalid file extension for market config: '{self.market_config}'. "
                f"Expected a '.json' file."
            )
        return self
    
    @model_validator(mode="after")
    def adapt_paths_and_create_directory(self):
        """
        Adjusts and ensures the directory structure for flex file generation and results storage.

        This method:
        - Updates `flex_files_directory` and `results_directory` paths so they are relative to 
        the base flex directory, using only the directory names (ignoring any user-supplied paths).
        - Creates the base, flex files, and results directories if they do not already exist.
        """
        # adapt paths and use only names for user supplied data
        self.flex_files_directory = (
            self.flex_base_directory_path
            / self.flex_files_directory.name
        )
        self.results_directory = (
            self.flex_base_directory_path
            / self.results_directory.name
        )
        # create directories if not already existing
        self.flex_base_directory_path.mkdir(parents=True, exist_ok=True)
        self.flex_files_directory.mkdir(parents=True, exist_ok=True)
        self.results_directory.mkdir(parents=True, exist_ok=True)
        return self