import tensorflow as tf
import math
import time
import scipy
import sys
import statsmodels.api as sm
from keras.models import Model, load_model
import matplotlib.pyplot as plt
import pyswarms as ps

from anesthesia_models import *
from patient import Patient,Gender

def b(a,p):
    return tf.constant([tf.reshape(tf.concatenate((a[0,i+1:,0],p[:i+1])),[180,1]) for i in range(len(p))])

class NNET:
    def __init__(self,p):
        self.z = p.z
        self.mdl = load_model('./weights/')
        self.patient = p

    @tf.function
    def __call__(self,p,r):
        return self.mdl((p,r,self.z))[-1][-1]

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

@tf.function
def stateless_pid(k,last_err,err_i,y,ref,rho,prop,remi,pred):
    err = ref-y
    d = last_err-err
    i = err_i+((last_err+err)/2)
    action = tf.clip_by_value(tf.reshape(tf.matmul(tf.reshape([err,i,d],(1,3)),k),(1,1,1)),0,20)
    o = pred.mdl((tf.concat((prop[:,1:,:],action),axis=1),tf.concat((remi[:,1:,:],rho*action),axis=1),pred.z))
    return i,action,o

class PID:
    def __init__(self,mdl,p=1,i=1,d=1):
        self.rho = 0.5
        self.pred = mdl
        self.k = tf.cast(tf.reshape([p,i,d],(3,1)),dtype='float32')
        self.err = tf.Variable(0.0)
        self.err_i = tf.Variable(0.0)
        self.prop = tf.TensorArray(tf.float32,size=180,dynamic_size=False)
        self.remi = tf.TensorArray(tf.float32,size=180,dynamic_size=False)
        self.prop.unstack(tf.zeros(180))
        self.remi.unstack(tf.zeros(180))
        self.index = 0


    def update(self,ref,y):
        state = stateless_pid(self.k,self.err,self.err_i,y,ref,self.rho,tf.reshape(self.prop.stack(),(1,180,1)),tf.reshape(self.remi.stack(),(1,180,1)),self.pred)
        self.err_i.assign(state[0])
        self.prop = self.prop.write(self.index,state[1][0][0][0])
        self.remi = self.remi.write(self.index,state[1][0][0][0]*self.rho)
        self.index += 1
        return state[2][0][0] 
    @tf.function
    def __call__(self,ref,y):
        return self.update(ref,y)
class MPC:
    def __init__(self,patient: Patient,nnet: Model,horizon=10):
        self.horizon = horizon 
        self.patient = patient
        self.prop = tf.zeros([1,180,1])
        self.remi = tf.zeros([1,180,1])
        self.nnet = nnet
    def update(self,ref,x):
        p,r = self.gen_infusion(ref,x-self.nnet.predict((self.prop,self.remi,self.patient.z),verbose=None))
        p = tf.constant([p[0]])
        r = tf.constant([r[0]])
        prediction = self.predict(p,r)
        self.prop = tf.reshape(tf.concat((self.prop[0,p.shape[0]:,0],p)),[1,180,1])
        self.remi = tf.reshape(tf.concat((self.remi[0,r.shape[0]:,0],r)),[1,180,1])
        return prediction 
    
    def predict(self,p,r):
        prop_infusion = b(self.prop,p)
        remi_infusion = b(self.remi,r)
        return self.nnet.predict((prop_infusion,remi_infusion,tf.tile(self.patient.z,(p.shape[0],1))),verbose=None)

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
            return (a/(((a<tol)*1000)+1))+((tf.sqrt(sum(map(lambda y:y**2,x)))**2)*2)
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
    def tot_err(ctl,mdl,ref,n):
        gen = iter_pid(ctl,mdl,ref)
        return 1e5*sum([np.abs(next(gen)) for i in range(n)])
    def cost(x):
        c = PID(mdl.patient,mdl,x[0],x[1],x[1],0.5)
        return tot_err(c,mdl,ref,70)
    return scipy.optimize.minimize(cost,np.array([-1,0,0]),options={'maxiter':40,'eps':1e-6,'disp':False})

def swarm_PID(mdl,ref):
    @tf.function
    def tot_err(u,ref,n):
        last_err = 0.98
        i = 0.0
        y = 0.98
        s = 0.0
        prop = tf.TensorArray(tf.float32,size=180)
        prop.unstack(tf.zeros((180)))
        remi = tf.TensorArray(tf.float32,size=180)
        remi.unstack(tf.zeros((180)))
        for j in range(n):
            state = stateless_pid(u,last_err,i,y,ref,0.5,tf.reshape(prop.stack(),(1,180,1)),tf.reshape(remi.stack(),(1,180,1)),mdl)
            prop = prop.write(j,state[1][0][0][0])
            remi = remi.write(j,state[1][0][0][0]*0.5)
            last_err = ref-state[2][0][0]
            y = state[2][0][0]
            i = state[0]
            s += tf.abs(last_err)
        return 1e5*s
    @tf.function
    def cost(y):
        return tf.vectorized_map(lambda x: tot_err(tf.reshape([x[0],x[1],x[2]],(3,1)),tf.constant(0.5,tf.float32),tf.constant(70)),tf.cast(y,tf.float32))
    print(cost(tf.constant([[1.0,1.0,1.0]])))
    options = {'c1':0.5, 'c2':0.3, 'w':0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=5000, dimensions=3, options=options)
    return optimizer.optimize(cost,iters=1000)[1]
    
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

def test():
    p = Patient(56,160,60,Gender.F)
    #n = Pharmacodynamic(p,2.0321,2.3266,13.9395,26.6474) 
    n = NNET(p)
    #mpc = MPC(p,n.mdl,5)
    #x = n(np.zeros((1,180,1)),np.zeros((1,180,1)))
    #ys = [x]
    #for i in range(50):
    #    x = mpc.update(0.5,x)[-1][-1]
    #    ys.append(x)
    #    print(ys)
    st = time.time() 
    #pid,_ = fit_PID(n,0.5,ys)
    #c = PID(p,n,*pid)
    #fig,ax = plt.subplots(subplot_kw=dict(projection='3d'))
    #ax.plot_surface(*PID_plot(n))
    optimized_values = swarm_PID(n,0.5)
    c = PID(n,*optimized_values)
    #c = PID(p,n,-1.55470255,0, -0.00467335)
    print(time.time()-st,c.k)
    bis = [n(np.ones([1,180,1])*1e-16,np.ones([1,180,1])*1e-16)]*50
    #print(bis)
    true_bis = []
    rs = []
    at = time.time()
    refs = (tf.ones(15)*0.5)
    for a,r in enumerate(refs):
        for i in range(10):
            st = time.time()
            b = c.update(r,bis[-1])
            #bis.extend(b+(int(a>len(refs[int(len(refs)/2):])/2))*0.2+np.random.normal(scale=0.05))
            bis.append(b)
            rs.extend([r])
    plt.figure()
    plt.plot(bis[50:])
    print(c.prop.stack())
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
