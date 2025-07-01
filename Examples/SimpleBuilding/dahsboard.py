from agentlib_mpc.utils.plotting.interactive import show_dashboard
from agentlib_mpc.utils.analysis import load_mpc
from pathlib import Path 
import pandas as pd

def dashy(): 
    path1 = Path("D:/fse-tve/git/flexquant/Examples/SimpleBuilding/results") / "mpc_simple_building_base.csv"
    path2 = Path("D:/fse-tve/git/flexquant/Examples/SimpleBuilding/results") / "mpc_simple_building_pos_flex.csv"
    data1 = load_mpc(path1)
    data1["scenario"] = "base"
    data2 = load_mpc(path2)
    data2["scenario"] = "pos_flex"

    
    
    # data = data1

   
    show_dashboard(data = data2, scale = "hours")

    

if __name__ == '__main__':
    dashy()
    