import numpy as np
import time
import scipy
from keras.models import Model, load_model
import matplotlib.pyplot as plt

from patient import Patient,Gender

def b(a,p):
    return np.array([np.concatenate((a[0,i+1:,0],p[:i+1])).reshape([180,1]) for i in range(len(p))])

class MPC:
    def __init__(self,patient: Patient,nnet: Model,horizon=10):
        self.horizon = horizon 
        self.patient = patient
        self.prop = np.zeros([1,180,1])
        self.remi = np.zeros([1,180,1])
        self.nnet = nnet

    def update(self,ref,x):
        p,r = self.gen_infusion(ref,x-self.nnet.predict((self.prop,self.remi,self.patient.z),verbose=None))
        prediction = self.predict(p,r)
        self.prop = np.concatenate((self.prop[0,self.horizon:,0],p)).reshape([1,180,1])
        self.remi = np.concatenate((self.remi[0,self.horizon:,0],r)).reshape([1,180,1])
        return prediction
    
    def predict(self,p,r):
        prop_infusion = b(self.prop,p)
        remi_infusion = b(self.remi,r)
        return self.nnet.predict((prop_infusion,remi_infusion,np.tile(self.patient.z,(self.horizon,1))),verbose=None)

    def gen_infusion(self,ref,bias):
        ref -= bias 
        inputs = scipy.optimize.minimize(self.gen_cost(ref),np.ones(self.horizon*2),method='nelder-mead',options={'disp':True,'maxfev':self.horizon*100},bounds=[(0,20)],tol=1).x
        return inputs[:self.horizon],inputs[self.horizon:]

    def gen_cost(self,ref):
        return lambda x: sum((self.nnet.predict((b(self.prop,x[:self.horizon]),b(self.remi,x[self.horizon:]),np.tile(self.patient.z,(self.horizon,1))),verbose=None)-ref)**2)*10000



def test():
    p = Patient(30,180,60,Gender.F)
    nnet = load_model('./weights')
    mpc = MPC(p,nnet,5)
    bis = list(nnet.predict((np.zeros([1,180,1]),np.zeros([1,180,1]),p.z)))
    at = time.time()
    for i in range(20):
        st = time.time()
        b = mpc.update(0.5,bis[-1])
        bis.extend(b+0.05)
        print([x[0] for x in bis[-mpc.horizon:]],time.time()-st)
    print(time.time()-at)
    plt.figure()
    plt.plot(bis)
