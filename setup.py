import setuptools

requirements = [
    "setuptools",
    "agentlib[full]==0.8.7",
    "agentlib_mpc[full] @ git+https://github.com/RWTH-EBC/AgentLib-MPC.git@849f87ddc674bedf24c2b98e92496f300b61da5a",
    "pathlib",
    "astor==0.8.1",
    "black",
    "pre-commit",
    "pytest",
    "pytest-snapshot"
]


setuptools.setup(
    name="flexquant",
    version="0.1.0",
    author="",
    author_email="",
    description="Flexibility quantification setup based on agentlib_mpc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements
)
