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

@tf.function
def sequential_append(a,p):
    b = tf.reshape(a,(1,180))[0,:]
    return tf.map_fn(lambda i: tf.reshape(tf.concat((tf.gather(tf.roll(b,-tf.cast(i,tf.int64),axis=0),tf.range(b.shape[0]-i,0,delta=-1,dtype=tf.int64)),tf.gather(p,tf.range(i,dtype=tf.int64))),0),(a.shape[1],1)),tf.range(1,p.shape[0]+1,dtype=a.dtype))

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
    d = (err-last_err)
    i = tf.reshape(err_i+(err*1),())
    action = tf.clip_by_value(k[0]*(err+(1/k[1])*i+(k[2]/1)*d),0,np.inf)
    o = pred.mdl((tf.concat((prop[:,1:,:],tf.reshape(action,(1,1,1))),axis=1),tf.concat((remi[:,1:,:],rho*tf.reshape(action,(1,1,1))),axis=1),pred.z))
    return i,tf.reshape(action,(1,1,1)),o

class PID:
    def __init__(self,mdl,p=1,i=1,d=1):
        self.rho = 1
        self.pred = mdl
        self.k = tf.cast([p,i,d],dtype='float32')
        self.err = tf.Variable(0.5-0.98)
        self.err_i = tf.Variable(0.0)
        self.prop = tf.TensorArray(tf.float32,size=180,dynamic_size=False)
        self.remi = tf.TensorArray(tf.float32,size=180,dynamic_size=False)
        self.prop.unstack(tf.zeros(180))
        self.remi.unstack(tf.zeros(180))
        self.index = 0


    def update(self,ref,y):
        state = stateless_pid(self.k,self.err,self.err_i,y,ref,self.rho,tf.reshape(self.prop.stack(),(1,180,1)),tf.reshape(self.remi.stack(),(1,180,1)),self.pred)
        print(state)
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
        self.prop = tf.zeros([1,180,1],dtype=tf.float64)
        self.remi = tf.zeros([1,180,1],dtype=tf.float64)
        self.nnet = nnet

    def update(self,ref,x):
        p,r = self.gen_infusion(ref,x-self.nnet((self.prop,self.remi,self.patient.z)))
        self.prop = tf.reshape(tf.concat((tf.transpose(self.prop[0,:,0]),[p[0]]),axis=0)[1:],(1,180,1))
        self.remi = tf.reshape(tf.concat((tf.transpose(self.remi[0,:,0]),[r[0]]),axis=0)[1:],(1,180,1))
        return self.nnet((self.prop,self.remi,self.patient.z))[-1][-1]

    def gen_infusion(self,ref,bias):
        ref -= bias 
        maxiter = 1e6
        loss = lambda x: stateless_cost(self.nnet,self.patient.z,self.prop,self.remi,self.horizon,x,ref)
        j = lambda x: loss_jac(self.nnet,self.patient.z,self.prop,self.remi,self.horizon,x,ref)
        h = lambda x: loss_hess(self.nnet,self.patient.z,self.prop,self.remi,self.horizon,x,ref)
        inputs = scipy.optimize.minimize(loss,tf.ones(self.horizon*2),jac=j,method='L-BFGS-B',options={'disp':False}).x
        return inputs[:self.horizon],inputs[self.horizon:]
@tf.function
def loss_hess(nnet,z,prop,remi,horizon,x,ref):
    x = tf.cast(x,prop.dtype)
    a = 0.0
    for i in range(1,horizon):
        a += ((nnet((tf.reshape(tf.concat((prop[0,i:,0],x[:i]),axis=0),(1,180,1)),tf.reshape(tf.concat((remi[0,i:,0],x[horizon:horizon+i]),axis=0),(1,180,1)),z))[-1][-1]-ref)**2)*(100/(i+1))
    if not tf.math.reduce_all(tf.abs(x)==x):
        a *= 1e16     
    return tf.gradients(tf.gradients(a,x)[0],x)[0]

@tf.function
def loss_jac(nnet,z,prop,remi,horizon,x,ref):
    x = tf.cast(x,prop.dtype)
    a = 0.0
    for i in range(1,horizon):
        a += ((nnet((tf.reshape(tf.concat((prop[0,i:,0],x[:i]),axis=0),(1,180,1)),tf.reshape(tf.concat((remi[0,i:,0],x[horizon:horizon+i]),axis=0),(1,180,1)),z))[-1][-1]-ref)**2)*(1000/(i+1))
    if not tf.math.reduce_all(tf.abs(x)==x):
        a *= 1e16 
    return tf.gradients(a,x)[0]
@tf.function
def stateless_cost(nnet,z,prop,remi,horizon,x,ref):
    x = tf.cast(x,prop.dtype)
    a = 0.0
    for i in range(1,horizon):
        a += ((nnet((tf.reshape(tf.concat((prop[0,i:,0],x[:i]),axis=0),(1,180,1)),tf.reshape(tf.concat((remi[0,i:,0],x[horizon:horizon+i]),axis=0),(1,180,1)),z))[-1][-1]-ref)**2)*(1000/(i+1))
    if not tf.math.reduce_all(tf.abs(x)==x):
        a *= 1e16 
    return a

   
def lowpass(xs,dt,rc):
    a = dt/(rc+dt)
    y = [a*xs[0]]
    for i in range(1,len(xs)):
        y.append(a*xs[i] + (1-a)*y[i-1])
    return y

def get_smooth(xs):
    return lowpass(xs,1,3)

def get_PID(mdl,ref):
    @tf.function
    def tot_err(u,ref,n):
        last_err = 0.5-0.98
        i = 0.0
        y = 0.98
        s = 0.0
        prop = tf.TensorArray(tf.float32,size=180)
        prop.unstack(tf.zeros((180)))
        remi = tf.TensorArray(tf.float32,size=180)
        remi.unstack(tf.zeros((180)))
        for j in range(n):
            state = stateless_pid(u,last_err,i,y,ref,0.1,tf.reshape(prop.stack(),(1,180,1)),tf.reshape(remi.stack(),(1,180,1)),mdl)
            prop = prop.write(j,state[1][0][0][0])
            remi = remi.write(j,state[1][0][0][0]*0.1)
            last_err = ref-state[2][0][0]
            y = state[2][0][0]
            i = state[0]
            s += tf.abs(last_err)
        return 1e5*s
    @tf.function
    def cost(x):
        return tot_err(tf.cast(x,tf.float32),tf.cast(ref,tf.float32),tf.constant(90))
    @tf.function
    def j(x):
        a = tf.cast(x,tf.float32)
        return tf.gradients(tot_err(a,0.5,40),a)
    print(j(tf.constant([-1,0,0])))
    return scipy.optimize.minimize(cost,np.array([-1,0,0]),jac=j,options={'maxiter':100,'disp':True}).x

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
        for j in range(1,n):
            state = stateless_pid(u,last_err,i,y,ref,2.0,tf.reshape(prop.stack(),(1,180,1)),tf.reshape(remi.stack(),(1,180,1)),mdl)
            prop = prop.write(j,state[1][0][0][0])
            remi = remi.write(j,state[1][0][0][0]*2)
            last_err = ref-state[2][0][0]
            y = state[2][0][0]
            i = state[0]
            s += tf.abs(last_err)
        return 1e5*s
    @tf.function
    def cost(y):
        return tf.vectorized_map(lambda x: tot_err(tf.reshape([x[0],x[1],x[2]],(3,1)),tf.constant(0.5,tf.float32),tf.constant(70)),tf.cast(y,tf.float32))
    print(cost(tf.constant([[1.0,1.0,1.0]])))
    options = {'c1':5, 'c2':3, 'w':9}
    optimizer = ps.single.GlobalBestPSO(n_particles=5000, dimensions=3, options=options)
    return optimizer.optimize(cost,iters=50)[1]
    
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
    p = Patient(36,160,60,Gender.F)
    #n = Pharmacodynamic(p,2.0321,2.3266,13.9395,26.6474) 
    n = NNET(p)
    refs = (tf.ones(15)*0.5)

    c = MPC(p,n.mdl,5)
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
    #optimized_values = swarm_PID(n,0.5)
    #optimized_values =get_PID(n,0.5)
    #optimized_values =[-10,np.inf,0]
    #c = PID(n,*optimized_values)
    #c = PID(p,n,-1.55470255,0, -0.00467335)
    #print(time.time()-st,c.k)
    bis = [n(tf.ones([1,180,1])*1e-16,tf.ones([1,180,1])*1e-16)]*50
    #print(bis)
    true_bis = []
    rs = []
    at = time.time()
    for a,r in enumerate(refs):
        for i in range(10):
            st = time.time()
            b = c.update(r,bis[-1])
            print(b,c.prop[0,-1,0],c.remi[0,-1,0])
            #bis.extend(b+(int(a>len(refs[int(len(refs)/2):])/2))*0.2+np.random.normal(scale=0.05))
            bis.append(b)
            rs.extend([r])
    plt.figure()
    plt.plot(bis[50:])
    print(time.time()-at)
    #print(c.prop.stack())
