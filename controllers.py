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

    def update(self):
        p,r = self.gen_infusion()
        prediction = self.predict(p,r)
        self.prop = np.concatenate((self.prop[0,self.horizon:,0],p)).reshape([1,180,1])
        self.remi = np.concatenate((self.remi[0,self.horizon:,0],r)).reshape([1,180,1])
        return prediction
    
    def predict(self,p,r):
        prop_infusion = b(self.prop,p)
        remi_infusion = b(self.remi,r)
        return self.nnet.predict((prop_infusion,remi_infusion,np.tile(self.patient.z,(self.horizon,1))),verbose=None)

    def gen_infusion(self):
        inputs = scipy.optimize.minimize(self.gen_cost(0.5),np.ones(self.horizon*2),method='nelder-mead',options={'disp':True,'maxfev':self.horizon*100},bounds=[(0,20)],tol=1).x
        return inputs[:self.horizon],inputs[self.horizon:]

    def gen_cost(self,ref):
        return lambda x: sum((self.nnet.predict((b(self.prop,x[:self.horizon]),b(self.remi,x[self.horizon:]),np.tile(self.patient.z,(self.horizon,1))),verbose=None)-ref)**2)*10000
        #return lambda x: sum([((self.nnet.predict((np.concatenate((np.zeros([1,self.horizon-i,1]),self.prop,x[:self.horizon].reshape([1,self.horizon,1])[:,:i,:]),axis=1),np.concatenate((np.zeros([1,self.horizon-i,1]),self.remi,x[self.horizon:].reshape([1,self.horizon,1])[:,:i,:]),axis=1),self.patient.z),verbose=None)[0]-ref)*100)**2 for i in range(self.horizon)])



def test():
    p = Patient(50,170,60,Gender.F)
    nnet = load_model('./weights')
    mpc = MPC(p,nnet,5)
    bis = []
    for i in range(10):
        st = time.time()
        b = mpc.update()
        bis.extend(b)
        print(b,time.time()-st)

    plt.figure()
    plt.plot(bis)
