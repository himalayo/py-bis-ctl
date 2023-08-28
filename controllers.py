import numpy as np
import scipy
from keras.models import Model, load_model
import matplotlib.pyplot as plt

from patient import Patient,Gender

class MPC:
    def __init__(self,patient: Patient,nnet: Model,horizon=10):
        self.horizon = horizon 
        self.patient = patient
        self.prop = np.zeros([1,180-horizon,1])
        self.remi = np.zeros([1,180-horizon,1])
        self.nnet = nnet

    def update(self):
        p,r = self.gen_infusion()
        prediction = self.predict(p,r)
        self.prop = np.concatenate((self.prop[0,self.horizon:,0],p)).reshape([1,180-self.horizon,1])
        self.remi = np.concatenate((self.remi[0,self.horizon:,0],r)).reshape([1,180-self.horizon,1])
        return prediction
    
    def predict(self,p,r):
        prop_infusion = np.concatenate((self.prop,p.reshape([1,self.horizon,1])),axis=1)
        remi_infusion = np.concatenate((self.remi,r.reshape([1,self.horizon,1])),axis=1)
        return self.nnet.predict((prop_infusion,remi_infusion,self.patient.z),verbose=None)[0]

    def gen_infusion(self):
        inputs = scipy.optimize.minimize(self.gen_cost(0.5),np.ones(self.horizon*2),method='nelder-mead',options={'fatol':1e-8,'disp':True,'maxfev':self.horizon*100},bounds=[(0,None)]).x
        return inputs[:self.horizon],inputs[self.horizon:]

    def gen_cost(self,ref):
        return lambda x: ((self.nnet.predict((np.concatenate((self.prop,x[:self.horizon].reshape([1,self.horizon,1])),axis=1),np.concatenate((self.remi,x[self.horizon:].reshape([1,self.horizon,1])),axis=1),self.patient.z),verbose=None)[0]-ref)*100)**2



def test():
    p = Patient(50,170,60,Gender.F)
    nnet = load_model('./weights')
    mpc = MPC(p,nnet)
    bis = []
    for i in range(5):
        bis.append(mpc.update())
        print(bis[-1])
    plt.figure()
    plt.plot(bis)
    plt.figure()
    plt.plot(mpc.prop.reshape(170))
    plt.plot(mpc.remi.reshape(170))