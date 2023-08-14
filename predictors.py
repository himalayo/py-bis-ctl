import numpy as np
import control as ct
from patient import Patient
from keras.models import load_model

class Predictor:
    def __call__(self,u):
        return self.io(u)

class PKPD(Predictor):
    def __init__(self,beta,gamma,cp50,cr50):
        def hill(beta,gamma,cp50,cr50,prop,remi):
            u_prop = prop/cp50
            u_remi = remi/cr50
            phi = (u_prop)/(u_prop+u_remi)
            u50 = 1 - (beta*phi) + (beta*phi^2)
            return 97.7  * ( 1- ( ( (( u_prop + u_remi )/u50 )^gamma )/( 1 + ( ((u_prop + u_remi)/u50)^gamma) ) ) )
        self.io = ct.NonlinearIOSystem(lambda t,x,u,params=None: hill(beta,gamma,cp50,cr50,u[0],u[1]),inputs=2,states=1)

class Neural(Predictor):
    def __init__(self,path,patient: Patient):
        def state_update(t,x,u,params):
            params['prop'].append(x[0])
            params['remi'].append(x[1])
            return [u[0],u[1],params['nnet']([np.array(params['prop']).T,np.array(params['remi']).T,np.broadcast_to(params['clinical'],[len(params['prop']),4])])]
        self.nnet = load_model(path)
        self.params = {'prop': [], 'remi': [], 'clinical': patient.np}
        self.io = ct.NonlinearIOSystem(state_update,params=self.params,inputs=3,states=3)