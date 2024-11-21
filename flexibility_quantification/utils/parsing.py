import ast
from typing import Union, List, Optional
from flexibility_quantification.data_structures.mpcs import (
    BaseMPCData,
    PFMPCData,
    NFMPCData,
    BaselineMPCData,
)
from flexibility_quantification.data_structures.globals import (
    SHADOW_MPC_COST_FUNCTION,
    return_baseline_cost_function,
    full_trajectory_prefix,
    full_trajectory_suffix,
)
from agentlib_mpc.data_structures.mpc_datamodels import MPCVariable
from string import Template

# Constants
CASADI_INPUT = "CasadiInput"
CASADI_PARAMETER = "CasadiParameter"
CASADI_OUTPUT = "CasadiOutput"

# String templates
INPUT_TEMPLATE = Template(
    "$class_name(name='$name', value=$value, unit='$unit', description='$description')"
)
PARAMETER_TEMPLATE = Template(
    "$class_name(name='$name', value=$value, unit='$unit', description='$description')"
)
OUTPUT_TEMPLATE = Template(
    "$class_name(name='$name', unit='$unit', type='$type', value=$value, description='$description')"
)


def create_ast_element(template_string):
    return ast.parse(template_string).body[0].value


def add_input(name, value, unit, description, type):
    return create_ast_element(
        INPUT_TEMPLATE.substitute(
            class_name=CASADI_INPUT,
            name=name,
            value=value,
            unit=unit,
            description=description,
            type=type,
        )
    )


def add_parameter(name, value, unit, description):
    return create_ast_element(
        PARAMETER_TEMPLATE.substitute(
            class_name=CASADI_PARAMETER,
            name=name,
            value=value,
            unit=unit,
            description=description,
        )
    )


def add_output(name, unit, type, value, description):
    return create_ast_element(
        OUTPUT_TEMPLATE.substitute(
            class_name=CASADI_OUTPUT,
            name=name,
            unit=unit,
            type=type,
            value=value,
            description=description,
        )
    )


class SetupSystemModifier(ast.NodeTransformer):
    """A custom AST transformer for modifying the MPC model file.

    This class traverses the AST of the input file, identifies the relevant classes and methods,
    and performs the necessary modifications.

    Attributes:
        mpc_data (str): The new return expression to be used in the setup_system method.

    """

    def __init__(
        self,
        mpc_data: BaseMPCData,
        controls: List[MPCVariable],
        binary_controls: Optional[List[MPCVariable]],
    ):
        self.mpc_data = mpc_data
        self.controls = controls
        self.binary_controls = binary_controls
        # create object for ast parsing for both, the config and the model
        self.config_obj: Union[None, ast.expr] = None
        self.model_obj: Union[None, ast.expr] = None
        # select modification of setup_system based on mpc type
        if isinstance(mpc_data, (PFMPCData, NFMPCData)):
            self.modify_config_class = self.modify_config_class_shadow
            self.modify_setup_system = self.modify_setup_system_shadow
        if isinstance(mpc_data, BaselineMPCData):
            self.modify_config_class = self.modify_config_class_baseline
            self.modify_setup_system = self.modify_setup_system_baseline

    def visit_Module(self, module):
        """Visit a module definition in the AST.

        Appends or deletes the import statements at the top of the module.

        Args:
            module (ast.Module): The module definition node in the AST.

        Returns:
            ast.Module: The possibly modified module definition node.

        """
        # append imports for baseline
        if isinstance(self.mpc_data, BaselineMPCData):
            module = add_import_to_tree(name="pandas", alias="pd", tree=module)
            module = add_import_to_tree(name="casadi", alias="ca", tree=module)
        # delete imports for shadow MPCs
        if isinstance(self.mpc_data, (NFMPCData, PFMPCData)):
            module = remove_all_imports_from_tree(module)
        # trigger the next visit method (ClassDef)
        self.generic_visit(module)
        return module

    def visit_ClassDef(self, node):
        """Visit a class definition in the AST.

        This method is called for each class definition in the AST. It identifies the
        BaselineMPCModelConfig and BaselineMPCModel classes and performs the necessary actions.

        Args:
            node (ast.ClassDef): The class definition node in the AST.

        Returns:
            ast.ClassDef: The possibly modified class definition node.

        """
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "CasadiModelConfig":
                # get ast object and trigger modification
                self.config_obj = node
                self.modify_config_class(node)
                # change class name
                node.name = self.mpc_data.class_name + "Config"
            if isinstance(base, ast.Name) and base.id == "CasadiModel":
                # get ast object and trigger modification
                self.model_obj = node
                for item in node.body:
                    if (
                        isinstance(item, ast.FunctionDef)
                        and item.name == "setup_system"
                    ):
                        self.modify_setup_system(item)
                    # change config value
                    if isinstance(item, ast.AnnAssign) and item.target.id == "config":
                        item.annotation = (
                            ast.parse(self.mpc_data.class_name + "Config").body[0].value
                        )

                # change class name
                node.name = self.mpc_data.class_name

        return node

    def modify_config_class_shadow(self, node):
        """Modify the config class of the shadow mpc.

        Args:
            node (ast.ClassDef): The class definition node of the config.

        """
        # loop over config object and modify fields
        for body in node.body:
            # add the time and full control trajectory inputs
            if body.target.id == "inputs":
                body.value.elts.append(
                    add_input("Time", 0, "s", "time trajectory", "list")
                )
                for control in self.controls:
                    body.value.elts.append(
                        add_input(
                            f"{full_trajectory_prefix}{control.name}"
                            f"{full_trajectory_suffix}",
                            "pd.Series([0])",
                            "W",
                            "pd.Series",
                            "full control output",
                        )
                    )
                # also include binary controls
                if self.binary_controls:
                    for control in self.binary_controls:
                        body.value.elts.append(
                            add_input(
                                f"{full_trajectory_prefix}{control.name}"
                                f"{full_trajectory_suffix}",
                                "pd.Series([0])",
                                "W",
                                "full control output",
                                "pd.Series",
                            )
                        )
                body.value.elts.append(
                    add_input("in_provision", False, "-", "provision flag", "bool")
                )
            # add the flex variables and the weights
            if body.target.id == "parameters":
                for param_name in ["prep_time", "flex_event_duration", "market_time"]:
                    body.value.elts.append(
                        add_parameter(param_name, 0, "s", "time to switch objective")
                    )
                for weight in self.mpc_data.weights:
                    body.value.elts.append(
                        add_parameter(
                            weight.name,
                            weight.value,
                            "-",
                            "Weight for P in objective function",
                        )
                    )

    def modify_config_class_baseline(self, node):
        """Modify the config class of the baseline mpc.

        Args:
            node (ast.ClassDef): The class definition node of the config.

        """
        # loop over config object and modify fields
        for body in node.body:
            # add the fullcontrol trajectories to the baseline config class
            if body.target.id == "outputs":
                for control in self.controls:
                    body.value.elts.append(
                        add_output(
                            f"{full_trajectory_prefix}{control.name}"
                            f"{full_trajectory_suffix}",
                            "W",
                            "pd.Series",
                            "pd.Series([0])",
                            "full control output",
                        )
                    )
                # also include binary controls
                if self.binary_controls:
                    for control in self.binary_controls:
                        body.value.elts.append(
                            add_output(
                                f"{full_trajectory_prefix}{control.name}"
                                f"{full_trajectory_suffix}",
                                "W",
                                "pd.Series",
                                "pd.Series([0])",
                                "full control output",
                            )
                        )
            # add the flexibility inputs
            if body.target.id == "inputs":
                body.value.elts.append(
                    add_input("Time", 0, "s", "time trajectory", "list")
                )
                body.value.elts.append(
                    add_input(
                        "_P_external",
                        0,
                        "W",
                        "External power profile to be provided",
                        "pd.Series",
                    )
                )
                body.value.elts.append(
                    add_input(
                        "in_provision",
                        False,
                        "-",
                        "Flag signaling if the flexibility is in provision",
                        "bool",
                    )
                )
                body.value.elts.append(
                    add_input(
                        "rel_start",
                        0,
                        "s",
                        "relative start time of the flexibility event",
                        "int",
                    )
                )
                body.value.elts.append(
                    add_input(
                        "rel_end",
                        0,
                        "s",
                        "relative end time of the flexibility event",
                        "int",
                    )
                )

    def modify_setup_system_shadow(self, node):
        """Modify the setup_system method of the shadow mpc model class.

        This method changes the return statement of the setup_system method and adds
        all necessary new lines of code.

        Args:
            node (ast.FunctionDef): The function definition node of setup_system.

        """
        # constraint the control trajectories for t < market_time
        for i, item in enumerate(node.body):
            if (
                isinstance(item, ast.Assign)
                and isinstance(item.targets[0], ast.Attribute)
                and item.targets[0].attr == "constraints"
            ):
                if isinstance(item.value, ast.List):
                    for ind, control in enumerate(self.controls):
                        # insert control boundaries at beginning of function
                        node.body.insert(
                            0,
                            ast.parse(
                                f"{control.name}_upper = ca.if_else(self.Time.sym < self.market_time.sym, "
                                f"self.{full_trajectory_prefix}{control.name}{full_trajectory_suffix}.sym, "
                                f"self.{control.name}.ub)"
                            ).body[0],
                        )
                        node.body.insert(
                            0,
                            ast.parse(
                                f"{control.name}_lower = ca.if_else(self.Time.sym < self.market_time.sym, "
                                f"self.{full_trajectory_prefix}{control.name}{full_trajectory_suffix}.sym, "
                                f"self.{control.name}.lb)"
                            ).body[0],
                        )
                        # append to constraints
                        new_element = (
                            ast.parse(
                                f"({control.name}_lower, self.{control.name}, {control.name}_upper)"
                            )
                            .body[0]
                            .value
                        )
                        item.value.elts.append(new_element)
                    # also include binary controls
                    if self.binary_controls:
                        for ind, control in enumerate(self.binary_controls):
                            # insert control boundaries at beginning of function
                            node.body.insert(
                                0,
                                ast.parse(
                                    f"{control.name}_upper = ca.if_else(self.Time.sym < self.market_time.sym, "
                                    f"self.{full_trajectory_prefix}{control.name}{full_trajectory_suffix}.sym, "
                                    f"self.{control.name}.ub)"
                                ).body[0],
                            )
                            node.body.insert(
                                0,
                                ast.parse(
                                    f"{control.name}_lower = ca.if_else(self.Time.sym < self.market_time.sym, "
                                    f"self.{full_trajectory_prefix}{control.name}{full_trajectory_suffix}.sym, "
                                    f"self.{control.name}.lb)"
                                ).body[0],
                            )
                            # append to constraints
                            new_element = (
                                ast.parse(
                                    f"({control.name}_lower, self.{control.name}, {control.name}_upper)"
                                )
                                .body[0]
                                .value
                            )
                            item.value.elts.append(new_element)
                    break
        # loop through setup_system function to find return statement
        for i, stmt in enumerate(node.body):
            if isinstance(stmt, ast.Return):
                # store current return statement
                original_return = stmt.value
                new_body = [
                    # create new standard objective variable
                    ast.Assign(
                        targets=[ast.Name(id="obj_std", ctx=ast.Store())],
                        value=original_return,
                    ),
                    # create flex objective variable
                    ast.Assign(
                        targets=[ast.Name(id="obj_flex", ctx=ast.Store())],
                        value=ast.parse(
                            self.mpc_data.flex_cost_function, mode="eval"
                        ).body,
                    ),
                    # overwrite return statement with custom function
                    ast.Return(value=ast.parse(SHADOW_MPC_COST_FUNCTION).body[0].value),
                ]
                # append new variables to end of function
                node.body[i:] = new_body
                break

    def modify_setup_system_baseline(self, node):
        """Modify the setup_system method of the baseline mpc model class.

        This method changes the return statement of the setup_system method and adds
        all necessary new lines of code.

        Args:
            node (ast.FunctionDef): The function definition node of setup_system.

        """
        # set the control trajectories with the respective variables
        if self.binary_controls:
            controls_list = self.controls + self.binary_controls
        else:
            controls_list = self.controls
        full_traj_list = [
            ast.Assign(
                targets=[
                    ast.Attribute(
                        value=ast.Name(id="self", ctx=ast.Load()),
                        attr=f"{full_trajectory_prefix}{control.name}"
                        f"{full_trajectory_suffix}.alg",
                        ctx=ast.Store(),
                    )
                ],
                value=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=control.name,
                    ctx=ast.Load(),
                ),
            )
            for control in controls_list
        ]
        # loop through setup_system function to find return statement
        for i, stmt in enumerate(node.body):
            if isinstance(stmt, ast.Return):
                # store current return statement
                original_return = stmt.value
                new_body = [
                    # create new standard objective variable
                    ast.Assign(
                        targets=[ast.Name(id="obj_std", ctx=ast.Store())],
                        value=original_return,
                    ),
                    # overwrite return statement with custom function
                    ast.Return(
                        value=ast.parse(
                            return_baseline_cost_function(
                                profile_deviation_weight=self.mpc_data.profile_deviation_weight,
                                power_variable=self.mpc_data.power_variable,
                            )
                        )
                        .body[0]
                        .value
                    ),
                ]
                # append new variables to end of function
                node.body[i:] = full_traj_list + new_body
                break


def add_import_to_tree(name: str, alias: str, tree: ast.Module):
    import_statement = ast.Import(names=[ast.alias(name=name, asname=alias)])
    for node in tree.body:
        if isinstance(node, ast.Import):
            already_imported_names = [alias.name for alias in node.names]
            already_imported_alias = [alias.asname for alias in node.names]
            if (
                name not in already_imported_names
                and alias not in already_imported_alias
            ):
                tree.body.insert(0, import_statement)
            break
    else:
        tree.body.insert(0, import_statement)
    return tree


def remove_all_imports_from_tree(tree: ast.Module):
    # Create a new list to hold nodes that are not imports
    new_body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    # Update the body of the tree to the new list
    tree.body = new_body
    return tree
