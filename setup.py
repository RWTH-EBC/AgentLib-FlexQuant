import setuptools

requirements = [
    "setuptools",
    "agentlib[full]==0.8.7",
    "agentlib_mpc[fmu, interactive] @ git+https://github.com/RWTH-EBC/AgentLib-MPC.git@356fa23bb612667f176e3c14bd4e9330127e35e7",
    "pycombina @ git+https://github.com/adbuerger/pycombina.git",
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
