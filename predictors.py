import matplotlib.pyplot as plt
import numpy as np
import math
import control as ct
import time
from patient import Patient,Gender
from keras.models import load_model

mean_c =np.array([ 56.50928621,0.49630326,61.13655517,162.0409402])
std_c =np.array([15.38680592, 0.49998633, 9.23908289, 8.33116551])

class Predictor:
    def __call__(self,u: np.ndarray):
        st = time.time()
        out = ct.input_output_response(self.io,np.arange(u.shape[1]),u,X0=self.x0).outputs
        print(time.time()-st)
        return out

class PKPD(Predictor):
    def __init__(self,beta,gamma,cp50,cr50):
        def hill(beta,gamma,cp50,cr50,prop,remi):
            u_prop = prop/cp50
            u_remi = remi/cr50
            phi = (u_prop)/(u_prop+u_remi)
            u50 = 1 - (beta*phi) + (beta*phi**2)
            return 97.7  * ( 1- ( ( (( u_prop + u_remi )/u50 )**gamma )/( 1 + ( ((u_prop + u_remi)/u50)**gamma) ) ) )
        self.io = ct.NonlinearIOSystem(lambda t,x,u,params: x,lambda t,x,u,params=None: hill(beta,gamma,cp50,cr50,u[0],u[1]),inputs=2,states=2,outputs=1)
        self.x0=[0,0]

class Neural(Predictor):
    def __init__(self,path,patient: Patient):
        def state_update(t,x,u,params):
            def accum(seq):
                out = np.zeros([1,180,1])
                np.copyto(out[0,-min(math.ceil(len(seq)/10),180):,0],list(map(sum,np.array_split(seq,math.ceil(len(seq)/10))[-180:])))
                return out
            if t <= 1e-4:
                params['prop'] = np.array([u[0]/12])
                params['remi'] = np.array([u[1]/12])
                params['prop_input'] = accum(np.array([u[0]/12]))
                params['remi_input'] = accum(np.array([u[1]/12]))
                out = params['nnet'].predict((params['prop_input'],params['remi_input'],np.broadcast_to((params['patient'].np-mean_c)/std_c,[params['prop_input'].shape[0],4])),verbose=None)
                params['last_out'] = out[-1]
                params['last_t'] = t
                params['micro_prop'] = [u[0]]
                params['micro_remi'] = [u[1]]
                params['nnet'].reset_states()
            elif t-params['last_t'] >= 1:
                params['last_t'] = t
                params['micro_prop'].append(u[0])
                params['micro_remi'].append(u[1])
                params['prop'] = np.append(params['prop'],(sum(params['micro_prop'])/12))
                params['remi'] = np.append(params['remi'],(sum(params['micro_remi'])/12))
                params['prop_input'] = accum(params['prop'])
                params['remi_input'] = accum(params['remi'])
                out = params['nnet'].predict((params['prop_input'],params['remi_input'],np.broadcast_to((params['patient'].np-mean_c)/std_c,[params['prop_input'].shape[0],4])),verbose=None)
                params['last_out'] = out[-1]
                params['micro_prop'] = []
                params['micro_remi'] = []
            else:
                params['micro_prop'].append(u[0])
                params['micro_remi'].append(u[1])
            return params['last_out']-x

        self.nnet = load_model(path)
        self.x0 = self.nnet((np.zeros([1,180,1]),np.zeros([1,180,1]),((patient.np-mean_c)/std_c).reshape([1,4])))
        self.params = {'prop': [], 'remi': [], 't':-0.1,'patient': patient,'nnet':self.nnet}
        self.io = ct.NonlinearIOSystem(state_update,lambda t,x,u,params: x,params=self.params,inputs=2,states=1,outputs=1)

class NeuralStateless(Predictor):
    def __init__(self,path,patient: Patient):
        self.nnet = load_model(path)
        self.io = ct.NonlinearIOSystem(None,lambda t,x,u,params: self.nnet.predict([u[:180].reshape([1,180,1]),u[180:].reshape([1,180,1]),patient.z.reshape([1,4])]),None,inputs=360,outputs=1)


def test():
   plt.figure()
   plt.plot(PKPD(1,1.5,5,20)(np.array([np.sin(np.linspace(0,2*np.pi,1800))+1,np.sin(np.linspace(0,3*np.pi,1800))+1]))[0])
   plt.figure()
   plt.plot(Neural('./weights',Patient(50,170,70,Gender.F))(np.array([np.sin(np.linspace(0,2*np.pi,1800))*10+10,np.sin(np.linspace(0,3*np.pi,1800))*10+10]))[0])
   plt.figure()
   plt.plot(Neural('./weights',Patient(50,170,70,Gender.F))(np.ones([2,1800])*120)[0])

