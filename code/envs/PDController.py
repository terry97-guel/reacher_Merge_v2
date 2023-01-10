import numpy as np

class PD_Controller():
    def __init__(self, dt=0.1, Kp=0.01*60, Kd=0.001*60):
        self.start_point = np.zeros(shape=2) 
        self.dt           = dt 
        self.Kp           = Kp 
        self.Kd           = Kd 
        self.prev_error   = np.zeros(shape=2)
        
    def clear(self):
        self.prev_error   = np.zeros(shape=2)


    def control(self, diff_value):
        error             = -diff_value
        delta_err         = error-self.prev_error
        self.prev_error   = error
        
        PTerm = self.Kp * error
        DTerm = self.Kd * (delta_err)/self.dt
        output = PTerm + DTerm
        
        return output
            
            



