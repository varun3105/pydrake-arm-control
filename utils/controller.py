from pydrake.all import *
import numpy as np
import logging
from modern_robotics import IKinBody

class IKHandler():
    
    def __init__(self):

        self.home_oreintation = self._ComputeFK(np.zeros([7]))
        logging.info("[SUCCESS] END-EFFECTOR POSITION & ORIENTATION CALCULATED")
        self.screw_axis = self._getScrewAxis()
        logging.info("[SUCCESS] SCREW AXIS CALCULATED")
        self.eomg = 0.01
        self.ev = 0.001

    def _ComputeFK(self, angles):
        
        t1 = angles[0]
        T01 = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0.1564],
                [0, 0, 0, 1]
        ]) @ np.array([
            [np.cos(t1), -np.sin(t1), 0, 0],
            [np.sin(t1), np.cos(t1), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        t2 = angles[1]
        T12 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.0054],
            [0, 0, 1, 0.1284],
            [0, 0, 0, 1]
        ]) @ np.array([
            [np.cos(t2), 0, np.sin(t2), 0],
            [0, 1, 0, 0],
            [-np.sin(t2), 0, np.cos(t2), 0],
            [0, 0, 0, 1]
        ])
        
        t3 = angles[2]
        T23 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.0064],
            [0, 0, 1, 0.2104],
            [0, 0, 0, 1]
        ]) @ np.array([
            [np.cos(t3), -np.sin(t3), 0, 0],
            [np.sin(t3), np.cos(t3), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        t4 = angles[3]
        T34 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.0064],
            [0, 0, 1, 0.2104],
            [0, 0, 0, 1]
        ]) @ np.array([
            [np.cos(t4), 0, np.sin(t4), 0],
            [0, 1, 0, 0],
            [-np.sin(t4), 0, np.cos(t4), 0],
            [0, 0, 0, 1]
        ])
        
        t5 = angles[4]
        T45 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.0064],
            [0, 0, 1, 0.2084],
            [0, 0, 0, 1]
        ]) @ np.array([
            [np.cos(t5), -np.sin(t5), 0, 0],
            [np.sin(t5), np.cos(t5), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        t6 = angles[5]
        T56 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.1059],
            [0, 0, 0, 1]
        ]) @ np.array([
            [np.cos(t6), 0, np.sin(t6), 0],
            [0, 1, 0, 0],
            [-np.sin(t6), 0, np.cos(t6), 0],
            [0, 0, 0, 1]
        ])
        
        t7 = angles[6]
        T67 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.1059],
            [0, 0, 0, 1]
        ]) @ np.array([
            [np.cos(t7), -np.sin(t7), 0, 0],
            [np.sin(t7), np.cos(t7), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        T7 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.0615],
            [0, 0, 0, 1]
        ])
        
        H = T01@T12@T23@T34@T45@T56@T67@T7
        
        return H
    
    def _getScrewAxis(self):
        
        omega1 = np.array([0, 0, -1])
        omega2 = np.array([0, 1, 0])
        omega3 = np.array([0, 0, -1])
        omega4 = np.array([0, 1, 0])
        omega5 = np.array([0, 0, -1])
        omega6 = np.array([0, 1, 0])
        omega7 = np.array([0, 0, -1])

        # Points on the axes
        q1 = np.array([0, 0.0246, -1.0309])
        q2 = np.array([0, 0.0192, -0.9025])
        q3 = np.array([0, 0.0128, -0.6921])
        q4 = np.array([0, 0.0064, -0.4817])
        q5 = np.array([0, 0, -0.2733])
        q6 = np.array([0, 0, -0.1674])
        q7 = np.array([0, 0, -0.0615])

        # Calculate v vectors
        v1 = -np.cross(omega1, q1)
        v2 = -np.cross(omega2, q2)
        v3 = -np.cross(omega3, q3)
        v4 = -np.cross(omega4, q4)
        v5 = -np.cross(omega5, q5)
        v6 = -np.cross(omega6, q6)
        v7 = -np.cross(omega7, q7)

        # Construct screw axes
        S1 = np.concatenate((omega1, v1))
        S2 = np.concatenate((omega2, v2))
        S3 = np.concatenate((omega3, v3))
        S4 = np.concatenate((omega4, v4))
        S5 = np.concatenate((omega5, v5))
        S6 = np.concatenate((omega6, v6))
        S7 = np.concatenate((omega7, v7))

        # Combine screw axes into Blist
        Blist = np.array([S1, S2, S3, S4, S5, S6, S7]).T
        # Blist = np.array([S7, S6, S5, S4, S3, S2, S1]).T
        
        return Blist

    def _getGoalMatrix(self, position, orientation):
        
        alpha = orientation[0]
        Mx = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)]
        ])

        beta = orientation[1]
        My = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [np.sin(beta), 0, np.cos(beta)]
        ])

        gamma = orientation[2]
        Mz = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])

        R = Mx @ My @ Mz

        p = position

        T_des = np.array([
            [R[0][0], R[0][1], R[0][2], p[0]],
            [R[1][0], R[1][1], R[1][2], p[1]],
            [R[2][0], R[2][1], R[2][2], p[2]],
            [0, 0, 0, 1]
        ])

        return T_des

    def _adjustValues(self, angles):
        
        values = angles % (2*np.pi)
        
        arr = np.zeros([len(values)])
        
        for i in range(len(values)):
            if(values[i] < -np.pi):
                arr[i] = np.pi - (np.abs(values[i]) - np.pi)
            elif(values[i] > np.pi):
                arr[i] = -np.pi + (values[i] - np.pi)
            else:
                arr[i] = values[i]
        return arr
    
    def getIK(self, position, orientation, guess):
        
        goal_matrix = self._getGoalMatrix(position, orientation)
        x, success = IKinBody(self.screw_axis, self.home_oreintation, goal_matrix, guess, self.eomg, self.ev)
        angles = self._adjustValues(x)
        
        return angles, success

class Controller(LeafSystem):
    
    def __init__(self, plant):
        LeafSystem.__init__(self)
        
        self.DeclareDiscreteState(1)
        self.DeclarePeriodicDiscreteUpdateNoHandler(0.01)
        self.plant = plant
        
        ## Declare all input and output ports from force controller
        self.DeclareVectorInputPort("Feedback Left", BasicVector(14))
        self.DeclareVectorInputPort("Feedback Right", BasicVector(14))
        self.DeclareAbstractInputPort("Contact Feedback",
                                      AbstractValue.Make(ContactResults())) 
        self.DeclareVectorOutputPort("Current States", BasicVector(28), self.forwardState)
        self.DeclareVectorOutputPort("Goal States", BasicVector(28), self.forwardGoal)
       
       ## Defines an example trajectory
        self.trajectory_left = {
            0 : [[0.5, 0.5, 0.5], [0, 0, 0]],
            1 : [[0.5, 0.5, 0.5], [0.1, 0, 0]],
            2 : [[0.5, 0.5, 0.5], [0.2, 0, 0]],
            3 : [[0.5, 0.5, 0.5], [0.3, 0, 0]],
            4 : [[0.5, 0.5, 0.5], [0.4, 0, 0]]
        }
        
        self.trajectory_idx_left = 0
        
        self.trajectory_right = {
            0 : [[0.5, 0.5, 0.5], [0, 0, 0]],
            1 : [[0.5, 0.5, 0.5], [0.1, 0, 0]],
            2 : [[0.5, 0.5, 0.5], [0.2, 0, 0]],
            3 : [[0.5, 0.5, 0.5], [0.3, 0, 0]],
            4 : [[0.5, 0.5, 0.5], [0.4, 0, 0]]
        }
        
        self.trajectory_idx_right = 0
        
        self.ik_handler = IKHandler()
        
        left, success = self.ik_handler.getIK(
                    self.trajectory_left[self.trajectory_idx_left][0],
                    self.trajectory_left[self.trajectory_idx_left][1],
                    np.zeros([7])
                )
        
        right, success = self.ik_handler.getIK(
                    self.trajectory_right[self.trajectory_idx_right][0],
                    self.trajectory_right[self.trajectory_idx_right][1],
                    np.zeros([7])
                )
        
        self.actuator_goal_left = left
        self.actuator_goal_right = right

    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        
        feedback_left = self.get_input_port(0).Eval(context)[0:7]
        feedback_right = self.get_input_port(0).Eval(context)[0:7]
        
        error_range = 0.05
        
        goals_complete_left = False
        goals_complete_right = False
        
        ## Runs this if last point of trajectory is reached
        if(self.trajectory_idx_left == len(self.trajectory_left)):
            goals_complete_left = True
        if(self.trajectory_idx_right == len(self.trajectory_right)):
            goals_complete_right = True
        
        ## Updates the goal of the trajectory points
        if((np.all(np.abs(self.actuator_goal_left-feedback_left) <= error_range)) and not goals_complete_left ):
            logging.info("[SUCCESS] REACHED GOAL, SETTING NEW GOAL [LEFT]")
            self.trajectory_idx_left +=1
            x, success = self.ik_handler.getIK(
                    self.trajectory_left[self.trajectory_idx_left][0],
                    self.trajectory_left[self.trajectory_idx_left][1],
                    feedback_left
                )
            self.actuator_goal_left = x
        
        if((np.all(np.abs(self.actuator_goal_right-feedback_right) <= error_range)) and not goals_complete_right ):
            logging.info("[SUCCESS] REACHED GOAL, SETTING NEW GOAL [RIGHT]")
            self.trajectory_idx_right +=1
            x, success = self.ik_handler.getIK(
                    self.trajectory_right[self.trajectory_idx_right][0],
                    self.trajectory_right[self.trajectory_idx_right][1],
                    feedback_right
                )
            self.actuator_goal_right = x
            
    def forwardState(self, context, output): 
        
        ## Forward the current arm states to the inverse dynamics controller
        feedback_left = self.get_input_port(0).Eval(context)
        feedback_right = self.get_input_port(1).Eval(context)
        states = np.concatenate((feedback_left[0:7], feedback_right[0:7], feedback_left[7:14], feedback_right[7:14]))
        output.SetFromVector(states)
        
    def forwardGoal(self, context, output):
        
        ## Forward the current goal state to the inverse dynamics controller
        actuator_values = np.concatenate((self.actuator_goal_left, self.actuator_goal_right, np.zeros([7]), np.zeros([7])))
        output.SetFromVector(actuator_values)
