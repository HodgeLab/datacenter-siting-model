# datacenter-siting-model
The datacenter siting model for capacity expansion, interconnection with fiber, the grid, and resource adequacy considerations on a county level for the US.

The data sources and description of data processing is described in the Base_Optimization.pdf file

## Environment 
The packages and versions used are in siting_env.yml

## Code
1. To get the data loaded and in the right format --> data_loader.py
   
2. The capacity expansion optimization model --> siting_model.py
   
3a. To run the model --> run_optimization.py

3b. To run the model and get some visualizations --> run_visualization.py

4. For setting up sweep of different parameters --> sweeps.py
   
5. To run the sweeps --> run_sweeps.py

6. After running the sweep, to visualize sweep results --> visualize_sweeps.py

