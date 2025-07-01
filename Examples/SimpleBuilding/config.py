from pathlib import Path


class PathVariables(): 

    flex_configs: Path = Path(__file__).parent / "flex_configs"
    mpc_configs: Path = Path(__file__).parent / "mpc_and_sim"
    predictor_configs: Path = Path(__file__).parent / "predictor"

    def __init__(): 
        """
        This class is used to store the paths to the configuration files.
        It is a singleton class, so it should not be instantiated.

        paths = PathVariables()
        """
        raise NotImplementedError("This class is a singleton and cannot be instantiated.")

    def this_is_a_function(self):
        pass 
    
    @staticmethod
    def this_is__static_function(a:int, b:int):
        c = a+b 
        return c

    @classmethod # wichtig, um folgende Notation zu verwenden : 
    def get_some_irrelevant_path(cls, parameter: str) -> Path:
        """
        This method is just an example to show how to use the class.
        It returns a path based on the given parameter.

        from config import PathVariables 

        path_to_something = PathVariables.get_some_irrelevant_path("example_parameter")
        """
        return cls.flex_configs / f"some_irrelevant_path_{parameter}.json"