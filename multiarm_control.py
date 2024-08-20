from pydrake.all import *
import numpy as np
import logging
from modern_robotics import IKinBody
from utils.controller import Controller, IKHandler
from utils.diagram import plantSetup, flowSetup

def main():
    ## Perform basic initialization
    builder = DiagramBuilder()
    meshcat = StartMeshcat()
    
    ## Create two multibody plants and two scene graphs
    plant, scene_graph, elements = plantSetup(builder, False)
    plant_vis, scene_graph_vis, elements_vis = plantSetup(builder, True)
    
    controller = Controller(plant) ## Define force controller
    
    ## Build the computation system and visualisation system
    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph_vis, meshcat)
    diagram, system_component = flowSetup(builder, controller, plant, plant_vis, elements_vis)
    
    ## Declares system context
    root_context = diagram.CreateDefaultContext()
    system_context = diagram.GetMutableSubsystemContext(system_component, root_context)
    plant_context = diagram.GetMutableSubsystemContext(plant, root_context)
    
    ## Initialize the robot start position for the mathematical plant (not the visualisation plant)
    plant.get_actuation_input_port().FixValue(plant_context, np.zeros(14))
        
    logging.info("[EXEC] Executing path...")
    
    ## Runs all the simulations
    simulator = Simulator(diagram, root_context)
    simulator.set_target_realtime_rate(1)
    simulator.Initialize()
    visualizer.StartRecording()
    simulator.AdvanceTo(15)
    visualizer.StopRecording()
    visualizer.PublishRecording()
    
    logging.info("[SUCCESS] PATH COMPLETE")
    
if __name__== "__main__":
    
    ## Setup logging
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler()  # Log to console
                        ])
    
    main() ## Runs the main function