from pydrake.all import *
import numpy as np
import logging
from modern_robotics import IKinBody

def plantSetup(builder, visualisation):
    logging.info("[EXEC] Building plant...")
    
    ## Create a plant and scene graph
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant, scene_graph)
    
    ## Load the left arm
    left_arm_urdf = "models/kinova_gen3/urdf/GEN3_URDF_V12.urdf"
    left_arm = parser.AddModelFromFile(left_arm_urdf, "left_arm")
    X_robot = RigidTransform()
    X_robot.set_translation([0,0,0.015])  
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("base_link", left_arm),
                     X_robot)
    
    ## Load the right arm
    right_arm_urdf = "models/kinova_gen3/urdf/GEN3_URDF_V12_2.urdf"
    right_arm = parser.AddModelFromFile(right_arm_urdf, "right_arm")
    X_robot2 = RigidTransform()
    X_robot2.set_translation([2, 0, 0.015])
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("base_link_2", right_arm),
                     X_robot2)

    ## Load the left gripper
    left_gripper = "models/fingers/left_finger.sdf"
    gripper = parser.AddModelFromFile(left_gripper, "left_gripper")
    X_grip = RigidTransform()
    X_grip.set_rotation(RotationMatrix(RollPitchYaw([0,np.pi/2,0]))) 
    plant.WeldFrames(plant.GetFrameByName("end_effector_link", left_arm),
                     plant.GetFrameByName("left_link", gripper),
                     X_grip)
    
    ## Load the right gripper
    right_gripper = "models/fingers/right_finger.sdf"
    gripper2 = parser.AddModelFromFile(right_gripper, "right_gripper")
    X_grip2 = RigidTransform()
    X_grip2.set_rotation(RotationMatrix(RollPitchYaw([0,np.pi/2,0])))
    plant.WeldFrames(plant.GetFrameByName("end_effector_link_2", right_arm),
                     plant.GetFrameByName("right_link", gripper2),
                     X_grip2)
    
    ## Define collisions in left arm
    gripper_bodies = [plant.GetBodyByName("left_link")]
    arm_bodies = [plant.GetBodyByName("bracelet_with_vision_link"),
                  plant.GetBodyByName("spherical_wrist_2_link"),
                  plant.GetBodyByName("spherical_wrist_1_link"),
                  plant.GetBodyByName("forearm_link"),
                  plant.GetBodyByName("half_arm_1_link"),
                  plant.GetBodyByName("half_arm_2_link"),
                  plant.GetBodyByName("shoulder_link"),
                  plant.GetBodyByName("base_link")]    
    gripper_ids = [plant.GetCollisionGeometriesForBody(body)[0] for body in gripper_bodies]
    arm_ids = [plant.GetCollisionGeometriesForBody(body)[0] for body in arm_bodies]
    set_one = GeometrySet(gripper_ids)
    set_two = GeometrySet(arm_ids)
    collision_filter = CollisionFilterDeclaration()
    collision_filter.ExcludeBetween(set_one, set_two)
    scene_graph.collision_filter_manager().Apply(collision_filter)
    
    ## Define collisions in right arm
    gripper_bodies = [plant.GetBodyByName("right_link")]
    arm_bodies = [plant.GetBodyByName("bracelet_with_vision_link_2"),
                  plant.GetBodyByName("spherical_wrist_2_link_2"),
                  plant.GetBodyByName("spherical_wrist_1_link_2"),
                  plant.GetBodyByName("forearm_link_2"),
                  plant.GetBodyByName("half_arm_1_link_2"),
                  plant.GetBodyByName("half_arm_2_link_2"),
                  plant.GetBodyByName("shoulder_link_2"),
                  plant.GetBodyByName("base_link_2")]
    gripper_ids = [plant.GetCollisionGeometriesForBody(body)[0] for body in gripper_bodies]
    arm_ids = [plant.GetCollisionGeometriesForBody(body)[0] for body in arm_bodies]
    set_one = GeometrySet(gripper_ids)
    set_two = GeometrySet(arm_ids)
    collision_filter = CollisionFilterDeclaration()
    collision_filter.ExcludeBetween(set_one, set_two)
    scene_graph.collision_filter_manager().Apply(collision_filter)
    
    if visualisation:
        ## Add a box to the world if visualisation is true
        parser.AddModelFromFile("models/box/box.sdf", "box_type")
        box_planar_joint = plant.AddJoint(
            PlanarJoint("cube_joint", plant.world_frame(), plant.GetFrameByName("cube_link"), np.array([0, 0, 0])))  
        box_planar_joint.set_default_translation([2,2])
        box_index = plant.GetModelInstanceByName("box_type")
        
        scene_graph.set_name("scene_graph_vis")
        plant.set_name("plant_vis")
    
    else:
        ## No box is defined here, only for the mathematical model
        scene_graph.set_name("scene_graph")
        plant.set_name("plant")
        box_index = 0
    
    ## Finalize the plant and return the contexts of each arm
    plant.Finalize()
    left_arm_index = plant.GetModelInstanceByName("left_arm")               ## left arm 
    right_arm_index = plant.GetModelInstanceByName("right_arm")             ## right arm
    
    elements = [left_arm_index, right_arm_index, box_index]
    
    logging.info("[SUCCESS] PLANT SETUP COMPLETE")
    
    return plant, scene_graph, elements

def flowSetup(builder, controller, plant, plant_vis, elements):
    logging.info("[EXEC] Building the diagram...")
        
    ## Defines the PID constant values for each of the joint actuators
    ## Different joints usually also have different values, but here it is taken to be the same
    dim = plant.num_actuators()
    kp = np.array([1] * dim)
    ki = np.array([0] * dim)
    kd = np.array([1.25] * dim)
    
    ## Defines all connections in the multibody systems
    ## Refer to the README.md for more information
    force_controller = builder.AddSystem(controller)
    zoh = builder.AddSystem(ZeroOrderHold(0.001, AbstractValue.Make(ContactResults())))
    inverse_dynamics_controller = builder.AddSystem(
        InverseDynamicsController(plant, kp, ki, kd, has_reference_acceleration=False))
    builder.Connect(plant_vis.get_state_output_port(elements[0]), force_controller.get_input_port(0))
    builder.Connect(plant_vis.get_state_output_port(elements[1]), force_controller.get_input_port(1))
    builder.Connect(plant_vis.get_contact_results_output_port(), zoh.get_input_port(0))
    builder.Connect(zoh.get_output_port(0), force_controller.get_input_port(2))
    builder.Connect(force_controller.get_output_port(0), inverse_dynamics_controller.get_input_port_estimated_state()) ## gets the state
    builder.Connect(force_controller.get_output_port(1), inverse_dynamics_controller.get_input_port_desired_state())
    builder.Connect(inverse_dynamics_controller.get_output_port_control(), plant_vis.get_actuation_input_port())
    
    ## Conclude the building for all three
    diagram = builder.Build()
    logging.info("[SUCCESS] FLOW SETUP COMPLETE")
    return diagram, inverse_dynamics_controller
