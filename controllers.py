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
        p = np.array([p[0]])
        r = np.array([r[0]])
        prediction = self.predict(p,r)
        self.prop = np.concatenate((self.prop[0,p.shape[0]:,0],p)).reshape([1,180,1])
        self.remi = np.concatenate((self.remi[0,r.shape[0]:,0],r)).reshape([1,180,1])
        return prediction 
    
    def predict(self,p,r):
        prop_infusion = b(self.prop,p)
        remi_infusion = b(self.remi,r)
        return self.nnet.predict((prop_infusion,remi_infusion,np.tile(self.patient.z,(p.shape[0],1))),verbose=None)

    def gen_infusion(self,ref,bias):
        ref -= bias 
        t = time.time()
        self.gen_cost(50,8*self.horizon)(np.ones(self.horizon*2))
        maxiter = (8)/(time.time()-t)
        inputs = scipy.optimize.minimize(self.gen_cost(ref,8*self.horizon),np.ones(self.horizon*2),method='nelder-mead',options={'maxfev':maxiter,'fatol':40,'disp':True},bounds=[(0,60)],tol=self.horizon*8).x
        return inputs[:self.horizon],inputs[self.horizon:]

    def gen_cost(self,ref,tol):
        c = lambda x:(sum((self.nnet.predict((b(self.prop,x[:self.horizon]),b(self.remi,x[self.horizon:]),np.tile(self.patient.z,(self.horizon,1))),verbose=None)-ref)**2)*100000)
        def cost(x):
            a=c(x)
            return (a/(((a<tol)*1000)+1))+((np.sqrt(sum(map(lambda y:y**2,x)))**2)*2)
        return cost 



def test():
    p = Patient(30,180,60,Gender.F)
    nnet = load_model('./weights')
    mpc = MPC(p,nnet,5)
    bis = list(nnet.predict((np.zeros([1,180,1]),np.zeros([1,180,1]),p.z)))
    rs = []
    at = time.time()
    refs = np.concatenate((np.ones(5)*0.5,np.ones(5)*0.7))
    for r in refs[:int(len(refs)/2)]:
        for i in range(10):
            st = time.time()
            b = mpc.update(r,bis[-1])
            bis.extend(b)
            rs.extend([r]*len(b))
            print([x for x in bis[-mpc.horizon:]],time.time()-st)
    for r in refs[int(len(refs)/2):]:
        for i in range(10):
            st = time.time()
            b = mpc.update(r,bis[-1])
            bis.extend(b+0.05)
            rs.extend([r]*len(b))
            print([x for x in bis[-mpc.horizon:]],time.time()-st)

    print(time.time()-at)
    plt.figure()
    plt.plot(list(map(lambda x:x*100,bis)),label="Simulated BIS (RNN)")
    plt.plot(list(map(lambda x:x*100,rs)),label="Reference")
    plt.xlabel("Time (10s)")
    plt.ylabel("BIS")
    plt.legend()
    plt.ylim(bottom=0)
    plt.figure()
    plt.plot(mpc.prop[0,180-len(bis):,0])
    plt.xlabel("Time (10s)")
    plt.ylabel("Propofol dose (ug)")
    plt.figure()
    plt.plot(mpc.remi[0,180-len(bis):,0])
    plt.xlabel("Time (10s)")
    plt.ylabel("Remifentanil dose (ng)")

