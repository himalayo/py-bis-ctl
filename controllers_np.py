import numpy as np
import math
import time
import scipy
import statsmodels.api as sm
from keras.models import Model, load_model
import matplotlib.pyplot as plt

from anesthesia_models import *
from patient import Patient,Gender

class NNET:
    def __init__(self,p):
        self.z = p.z
        self.mdl = load_model('./weights/')
        self.patient = p

    def __call__(self,p,r):
        return self.mdl.predict((p,r,self.z),verbose=None)[-1][-1]

class Pharmacodynamic:
    def __init__(self,patient,beta,gamma,cp50,cr50):
        self.patient = patient
        self.beta = beta
        self.gamma = gamma
        self.cp50=cp50
        self.cr50=cr50
    def hill(self,prop,remi):
            try:
                u_prop = prop/self.cp50
                u_remi = remi/self.cr50
                print(prop,remi)
                phi = (u_prop)/(u_prop+u_remi)
                if math.isnan(phi):
                    phi = 0
                u50 = 1 - (self.beta*phi) + (self.beta*phi**2)
                r= 97.7  * ( 1- ( ( (( u_prop + u_remi )/u50 )**self.gamma )/( 1 + ( ((u_prop + u_remi)/u50)**self.gamma) ) ) )
                return r
            except:
                return 97.7
    def __call__(self,p,r):
        return self.hill(p[0,-1,0],r[0,-1,0]) 



class PID:
    def __init__(self,patient,pred,p=1,d=1,i=1,rho=0.5):
        self.k = np.array([[p],[i],[d]])
        self.err = np.zeros(3)
        self.rho = rho
        self.pred=pred
        self.prop=np.zeros([1,180,1])
        self.remi=np.zeros([1,180,1])
    def update(self,ref,y):
        self.err[2] = (ref-y)-self.err[0]
        self.err[0] = ref-y
        self.err[1] += self.err[0]
        action = (self.err @ self.k).reshape((1,1,1))
        self.prop = np.copy(np.concatenate((np.copy(self.prop[:,1:,:]),action),axis=1))
        self.remi = np.copy(np.concatenate((np.copy(self.remi[:,1:,:]),self.rho*action),axis=1))
        return self.pred(self.prop,self.remi)

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
        #maxiter = (8)/(time.time()-t)
        maxiter = 1e6
        inputs = scipy.optimize.minimize(self.gen_cost(ref,8*self.horizon),np.ones(self.horizon*2),method='nelder-mead',options={'maxfev':maxiter,'fatol':self.horizon*8,'disp':True},bounds=[(0,60)],tol=self.horizon*8).x
        return inputs[:self.horizon],inputs[self.horizon:]

    def gen_cost(self,ref,tol):
        c = lambda x:(sum((self.nnet.predict((b(self.prop,x[:self.horizon]),b(self.remi,x[self.horizon:]),np.tile(self.patient.z,(self.horizon,1))),verbose=None)-ref)**2)*100000)
        def cost(x):
            a=c(x)
            return (a/(((a<tol)*1000)+1))+((np.sqrt(sum(map(lambda y:y**2,x)))**2)*2)
        return cost 


def lowpass(xs,dt,rc):
    a = dt/(rc+dt)
    y = [a*xs[0]]
    for i in range(1,len(xs)):
        y.append(a*xs[i] + (1-a)*y[i-1])
    return y

def get_smooth(xs):
    return lowpass(xs,1,3)

def get_PID(mdl,ref):
    def iter_pid(ctl,mdl,ref):
        ys = 0.98
        while True:
            ys = ctl.update(ref,ys)
            yield ref-ys
    def get_errs(ctl,mdl,ref):
        x = mdl(np.zeros([1,180,1]),np.zeros([1,180,1]))
        while True:
            x = ctl.update(ref,x)
            yield np.copy(ctl.err)
    def tot_err(ctl,mdl,ref,n):
        gen = iter_pid(ctl,mdl,ref)
        return 1e5*sum([np.abs(next(gen)) for i in range(n)])
    def cost(x):
        c = PID(mdl.patient,mdl,x[0],x[1],x[2],0.5)
        return tot_err(c,mdl,ref,70)
    def jac(x):
        cl = PID(mdl.patient,mdl,x[0],0,x[1],0.5)
        gen = get_errs(cl,mdl,ref)
        errs = np.array([next(gen) for i in range(70)])
        sum_errs = np.array((sum(errs[0,:]),sum(errs[1,:]),sum(errs[2,:])))
        sum_abs_errs = np.array((sum(np.abs(errs[0,:])),sum(np.abs(errs[1,:])),sum(np.abs(errs[2,:]))))
        out = [-np.abs(x[0]*sum_abs_errs[0])/(x[0]**3),(sum_errs[2]*np.abs(x[0])*np.abs(sum_abs_errs[0]))/(np.abs(x[0])*sum_errs[0]*x[0])]
        if math.isnan(out[0]):
            out[0] = 0
    def clbk(res):
        controller = PID(mdl.patient,mdl,res[0],res[1],res[2])
        gen = iter_pid(controller,mdl,ref)
        print(res,[next(gen) for i in range(50)])
    return scipy.optimize.minimize(cost,np.array([-1,0,0]),callback=clbk,options={'maxiter':80,'eps':1e-6,'disp':True})

def PID_plot(mdl):
    def iter_pid(ctl):
        x = 0.98
        while True:
            x = ctl.update(0.5,x)
            yield np.copy(0.5-x)
    def tot_err(ctl,n):
        return 1e2*sum([np.abs(next(iter_pid(ctl))) for i in range(n)])
    def cost(x):
        out = tot_err(PID(mdl.patient,mdl,x[0],x[1],x[2],0.5),50)
        c = PID(mdl.patient,mdl,x[0],x[1],x[2],0.5)
        print(x,out,[next(iter_pid(c)) for i in range(50)])
        return out
    xs = np.linspace(-3,3,10)
    ys = np.linspace(-3,3,10)
    X,Y = np.meshgrid(xs,ys)
    zs = np.array([[cost((x,0,y)) for x in xs] for y in ys])
    print(zs)
    return xs,ys,zs

if __name__=="__main__":
    p = Patient(56,160,60,Gender.F)
    #n = Pharmacodynamic(p,2.0321,2.3266,13.9395,26.6474) 
    n = NNET(p)
    #fig,ax = plt.subplots(subplot_kw=dict(projection='3d'))
    #ax.plot_surface(*PID_plot(n))
    st = time.time() 
    #optimized_values = get_PID(n,0.5)
    #print(time.time()-st,optimized_values)
    #c = PID(p,n,optimized_values.x[0],0,optimized_values.x[1])
    c = PID(p,n,-1.55470255,0, -0.00467335)
    bis = [n(np.ones([1,180,1])*1e-16,np.ones([1,180,1])*1e-16)]*50
    print(bis)
    true_bis = []
    rs = []
    at = time.time()
    refs = (np.ones(5)*0.5)
    for a,r in enumerate(refs):
        for i in range(10):
            st = time.time()
            b = c.update(r,bis[-1])
            print(b)
            #bis.extend(b+(int(a>len(refs[int(len(refs)/2):])/2))*0.2+np.random.normal(scale=0.05))
            bis.append(b)
            rs.extend([r])
    plt.figure()
    plt.plot(bis[50:])
    """
    p = Patient(30,180,60,Gender.F)
    nnet = load_model('./weights')
    mpc = MPC(p,nnet,5)
    bis = list(nnet.predict((np.zeros([1,180,1]),np.zeros([1,180,1]),p.z)))*50
    smoothed_bis = get_smooth(bis)
    true_bis = []
    print(smoothed_bis)
    rs = []
    at = time.time()
    refs = np.concatenate((np.ones(5)*0.5,np.ones(5)*0.7))
    for a,r in enumerate(refs[:int(len(refs)/2)]):
        for i in range(10):
            st = time.time()
            b = mpc.update(r,smoothed_bis[-1])
            #bis.extend(b+(int(a>len(refs[int(len(refs)/2):])/2))*0.2+np.random.normal(scale=0.05))
            bis.extend(b+np.random.normal(scale=0.05))
            smoothed_bis = get_smooth(bis)
            #true_bis.extend(b+(int(a>len(refs[int(len(refs)/2):])/2))*0.2)
            true_bis.extend(b)
            rs.extend([r]*len(b))
            print(smoothed_bis[-1],bis[-1],time.time()-st)
    for a,r in enumerate(refs[int(len(refs)/2):]):
        for i in range(10):
            st = time.time()
            b = mpc.update(r,smoothed_bis[-1])
            #bis.extend(b+(int(a>len(refs[int(len(refs)/2):])/2))*0.2+np.random.normal(scale=0.05))
            bis.extend(b+np.random.normal(scale=0.05))
            smoothed_bis = get_smooth(bis)
            #true_bis.extend(b+(int(a>len(refs[int(len(refs)/2):])/2))*0.2)
            true_bis.extend(b)
            rs.extend([r]*len(b))
            print(smoothed_bis[-1],bis[-1],time.time()-st)

    print(time.time()-at)
    plt.figure()
    plt.title("Com low-pass")
    #plt.plot(list(map(lambda x:x*100,smoothed_bis[50:])),label="Simulated BIS (smoothed)")
    plt.plot(list(map(lambda x:x*100,bis[50:])),label="Simulated BIS (noisy)")
    plt.plot(list(map(lambda x:x*100,true_bis)),label="Simulated BIS (true)")
    plt.plot(list(map(lambda x:x*100,rs)),label="Reference")
    plt.xlabel("Time (10s)")
    plt.ylabel("BIS")
    plt.legend()
    plt.figure()
    plt.title("Com low-pass")
    #plt.plot(list(map(lambda x:x*100,smoothed_bis[50:])),label="Simulated BIS (smoothed)")
    plt.plot(list(map(lambda x:x*100,bis[50:])),label="Simulated BIS (noisy)")
    #plt.plot(list(map(lambda x:x*100,true_bis)),label="Simulated BIS (true)")
    plt.plot(list(map(lambda x:x*100,rs)),label="Reference")
    plt.xlabel("Time (10s)")
    plt.ylabel("BIS")
    plt.figure()
    plt.title("Erro do lowpass")
    plt.plot(np.array(true_bis)-np.array(smoothed_bis[50:]),label="filter error")
    plt.legend()
    plt.legend()
    plt.ylim(bottom=0)lt.ylim(bottom=0)
    plt.figure()
    plt.plot(mpc.prop[0,180-len(bis)-50:,0])
    plt.xlabel("Time (10s)")
    plt.ylabel("Propofol dose (ug)")
    plt.figure()
    plt.plot(mpc.remi[0,180-len(bis)-50:,0])
    plt.xlabel("Time (10s)")
    plt.ylabel("Remifentanil dose (ng)")
    p = Patient(30,180,60,Gender.F)
    nnet = load_model('./weights')
    mpc = MPC(p,nnet,5)
    bis = list(nnet.predict((np.zeros([1,180,1]),np.zeros([1,180,1]),p.z)))*50
    smoothed_bis = get_smooth(bis)
    true_bis = [bis[-1]]
    print(smoothed_bis)
    rs = []
    at = time.time()
    refs = np.concatenate((np.ones(5)*0.5,np.ones(5)*0.7))
    for a,r in enumerate(refs[:int(len(refs)/2)]):
        for i in range(10):
            st = time.time()
            b = mpc.update(r,true_bis[-1])
            #bis.extend(b+(int(a>len(refs[int(len(refs)/2):])/2))*0.2+np.random.normal(scale=0.05))
            bis.extend(b+np.random.normal(scale=0.05))
            smoothed_bis = get_smooth(bis)
            #true_bis.extend(b+(int(a>len(refs[int(len(refs)/2):])/2))*0.2)
            true_bis.extend(b)
            rs.extend([r]*len(b))
            print(smoothed_bis[-1],bis[-1],time.time()-st)
    for a,r in enumerate(refs[int(len(refs)/2):]):
        for i in range(10):
            st = time.time()
            b = mpc.update(r,true_bis[-1])
            #bis.extend(b+(int(a>len(refs[int(len(refs)/2):])/2))*0.2+np.random.normal(scale=0.05))
            bis.extend(b+np.random.normal(scale=0.05))
            smoothed_bis = get_smooth(bis)
            #true_bis.extend(b+(int(a>len(refs[int(len(refs)/2):])/2))*0.2)
            true_bis.extend(b)
            rs.extend([r]*len(b))
            print(smoothed_bis[-1],bis[-1],time.time()-st)
    print(time.time()-at)
    plt.figure()
    plt.title("Filtro perfeito")
    #plt.plot(list(map(lambda x:x*100,smoothed_bis[50:])),label="Simulated BIS (smoothed)")
    plt.plot(list(map(lambda x:x*100,bis[50:])),label="Simulated BIS (noisy)")
    plt.plot(list(map(lambda x:x*100,true_bis)),label="Simulated BIS (true)")
    plt.plot(list(map(lambda x:x*100,rs)),label="Reference")
    plt.xlabel("Time (10s)")
    plt.ylabel("BIS")
    plt.legend()
    plt.figure()
    plt.title("Filtro perfeito")
    #plt.plot(list(map(lambda x:x*100,smoothed_bis[50:])),label="Simulated BIS (smoothed)")
    plt.plot(list(map(lambda x:x*100,bis[50:])),label="Simulated BIS (noisy)")
    #plt.plot(list(map(lambda x:x*100,true_bis)),label="Simulated BIS (true)")
    plt.plot(list(map(lambda x:x*100,rs)),label="Reference")
    plt.xlabel("Time (10s)")
    plt.ylabel("BIS")

    p = Patient(30,180,60,Gender.F)
    nnet = load_model('./weights')
    mpc = MPC(p,nnet,5)
    bis = list(nnet.predict((np.zeros([1,180,1]),np.zeros([1,180,1]),p.z)))*50
    smoothed_bis = get_smooth(bis)
    true_bis = []
    print(smoothed_bis)
    rs = []
    at = time.time()
    refs = np.concatenate((np.ones(5)*0.5,np.ones(5)*0.7))
    for a,r in enumerate(refs[:int(len(refs)/2)]):
        for i in range(10):
            st = time.time()
            b = mpc.update(r,bis[-1])
            #bis.extend(b+(int(a>len(refs[int(len(refs)/2):])/2))*0.2+np.random.normal(scale=0.05))
            bis.extend(b+np.random.normal(scale=0.05))
            smoothed_bis = get_smooth(bis)
            #true_bis.extend(b+(int(a>len(refs[int(len(refs)/2):])/2))*0.2)
            true_bis.extend(b)
            rs.extend([r]*len(b))
            print(smoothed_bis[-1],bis[-1],time.time()-st)
    for a,r in enumerate(refs[int(len(refs)/2):]):
        for i in range(10):
            st = time.time()
            b = mpc.update(r,bis[-1])
            #bis.extend(b+(int(a>len(refs[int(len(refs)/2):])/2))*0.2+np.random.normal(scale=0.05))
            bis.extend(b+np.random.normal(scale=0.05))
            smoothed_bis = get_smooth(bis)
            #true_bis.extend(b+(int(a>len(refs[int(len(refs)/2):])/2))*0.2)
            true_bis.extend(b)
            rs.extend([r]*len(b))
            print(smoothed_bis[-1],bis[-1],time.time()-st)
    print(time.time()-at)
    plt.figure()
    plt.title("Sem filtro")
    #plt.plot(list(map(lambda x:x*100,smoothed_bis[50:])),label="Simulated BIS (smoothed)")
    plt.plot(list(map(lambda x:x*100,bis[50:])),label="Simulated BIS (noisy)")
    #plt.plot(list(map(lambda x:x*100,true_bis)),label="Simulated BIS (true)")
    plt.plot(list(map(lambda x:x*100,rs)),label="Reference")
    plt.xlabel("Time (10s)")
    plt.ylabel("BIS")
    plt.legend()
    plt.figure()
    plt.title("Sem filtro")
    #plt.plot(list(map(lambda x:x*100,smoothed_bis[50:])),label="Simulated BIS (smoothed)")
    plt.plot(list(map(lambda x:x*100,bis[50:])),label="Simulated BIS (noisy)")
    plt.plot(list(map(lambda x:x*100,true_bis)),label="Simulated BIS (true)")
    plt.plot(list(map(lambda x:x*100,rs)),label="Reference")
    plt.xlabel("Time (10s)")
    plt.ylabel("BIS")
    plt.legend()
    """
