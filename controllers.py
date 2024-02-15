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
def stateless_pid(k,l_e,e_i,y,ref):
    e = ref-y
    d = (e-l_e)
    i = tf.reshape(e_i+(e+l_e)/2.0,())
    action = tf.tensordot([e,i,d],k,1)
    return action

class PID:
    def __init__(self,mdl,p=1,i=1,d=1,rho=1):
        self.rho =rho 
        self.pred = mdl
        self.k = tf.cast([p,i,d],dtype='float32')
        self.err = tf.Variable(0.5-0.98)
        self.err_i = tf.Variable(0.0)
        self.prop = tf.zeros((1,180,1))

    def update(self,ref,y):
        u = tf.clip_by_value(stateless_pid(self.k,self.err,self.err_i,y,ref),0,40)
        self.prop = tf.concat((self.prop[:,1:,:],[[[u]]]),axis=1)
        self.err_i.assign_add(((ref-y)+self.err)/2)
        self.err.assign(ref-y)
        return self.pred.mdl((self.prop,self.prop*self.rho,self.pred.patient.z))[-1][-1]
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
        loss = lambda x: mpc_loss(self.nnet,self.patient.z,self.prop,self.remi,self.horizon,x,ref)
        j = lambda x: mpc_loss_jac(self.nnet,self.patient.z,self.prop,self.remi,self.horizon,x,ref)
        h = lambda x: mpc_loss_hess(self.nnet,self.patient.z,self.prop,self.remi,self.horizon,x,ref)
        inputs = scipy.optimize.minimize(loss,tf.ones(self.horizon*2),jac=j,method='L-BFGS-B',options={'disp':False}).x
        return inputs[:self.horizon],inputs[self.horizon:]
@tf.function
def mpc_loss_hess(nnet,z,prop,remi,horizon,x,ref):
    x = tf.cast(x,prop.dtype)
    a = 0.0
    for i in range(1,horizon):
        a += ((nnet((tf.reshape(tf.concat((prop[0,i:,0],x[:i]),axis=0),(1,180,1)),tf.reshape(tf.concat((remi[0,i:,0],x[horizon:horizon+i]),axis=0),(1,180,1)),z))[-1][-1]-ref)**2)*(100/(i+1))
    if not tf.math.reduce_all(x>tf.ones(x.shape)*0.05) or not tf.math.reduce_all(x<=tf.ones(x.shape)*40):
        a *= 1e16      
    return tf.gradients(tf.gradients(a,x)[0],x)[0]

@tf.function
def mpc_loss_jac(nnet,z,prop,remi,horizon,x,ref):
    x = tf.cast(x,prop.dtype)
    a = 0.0
    for i in range(1,horizon):
        a += ((nnet((tf.reshape(tf.concat((prop[0,i:,0],x[:i]),axis=0),(1,180,1)),tf.reshape(tf.concat((remi[0,i:,0],x[horizon:horizon+i]),axis=0),(1,180,1)),z))[-1][-1]-ref)**2)*(100/(i+1))
    if not tf.math.reduce_all(x>tf.ones(x.shape,x.dtype)*0.05) or not tf.math.reduce_all(x<=tf.ones(x.shape,x.dtype)*40):
        a *= 1e16     
    return tf.gradients(a,x)[0]
@tf.function
def mpc_loss(nnet,z,prop,remi,horizon,x,ref):
    x = tf.cast(x,prop.dtype)
    a = 0.0
    for i in range(1,horizon):
        a += ((nnet((tf.reshape(tf.concat((prop[0,i:,0],x[:i]),axis=0),(1,180,1)),tf.reshape(tf.concat((remi[0,i:,0],x[horizon:horizon+i]),axis=0),(1,180,1)),z))[-1][-1]-ref)**2)*(100/(i+1))
    if not tf.math.reduce_all(x>tf.ones(x.shape,x.dtype)*0.05) or not tf.math.reduce_all(x<=tf.ones(x.shape,x.dtype)*40):
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

def get_PID(pred,ref,x0=tf.constant([-1.55,0,0.4])):
    @tf.function
    def gradient(kp,ki,kd,rho):
        f = loss(kp,ki,kd,rho,tf.constant(0.5),tf.constant(140),pred)
        grad = tf.gradients(f,[kp,ki,kd])
        tf.print([kp,ki,kd,rho],f,grad)
        return grad 
    return scipy.optimize.minimize(lambda x: tf.cast(pid_loss(*tf.cast(x,tf.float32),tf.constant(1.0),tf.constant(0.5),tf.constant(140),pred),tf.float64),x0,jac=lambda x: tf.cast(gradient(*tf.cast(x,tf.float32),1),tf.float64),method='L-BFGS-B',options={'disp':True}).x

def swarm_PID(pred,ref):
    @tf.function
    def cost(y):
        return tf.vectorized_map(lambda x: pid_loss(x[0],x[1],x[2],1,tf.constant(0.5,tf.float32),140,pred),tf.cast(y,tf.float32))
    options = {'c1':0.05, 'c2':0.03, 'w':0.09}
    optimizer = ps.single.GlobalBestPSO(n_particles=700, dimensions=3, options=options)
    return optimizer.optimize(cost,iters=50)[1]

@tf.function
def pid_loss(kp,ki,kd,rho,ref,n,pred):
        last_err = 0.5-0.98
        i = 0.0
        y = 0.98
        s = 0.0
        prop = tf.zeros((1,180,1))
        remi = tf.zeros((1,180,1))
        for j in range(n):
            err = ref-y
            i += (err+last_err)/2
            pu = kp*err+ki*i+kd*(err-last_err)
            ru = rho*(kp*err+ki*i+kd*(err-last_err))
            if pu<0:
                pu = tf.constant(0.0,pu.dtype)
            if pu>40:
                pu = tf.constant(40.0,pu.dtype)
            if ru<0:
                ru = tf.constant(0.0,ru.dtype)
            if ru>40:
                ru = tf.constant(40.0,ru.dtype)
            last_err = err 
            prop = tf.concat((prop[:,1:,:],[[[pu]]]),axis=1)
            remi = tf.concat((remi[:,1:,:],[[[ru]]]),axis=1)
            y = pred.mdl((prop,remi,pred.patient.z))[-1][-1]
            s += (last_err*10)**2
        return s   

def pid_loss_plot(pred,i):
    p = tf.cast(tf.linspace(-3,3,100),tf.float32)
    d = tf.cast(tf.linspace(-3,3,100),tf.float32)
    P,D = tf.meshgrid(p,d)

    @tf.function
    def cost(y):
        tf.print(y)
        return tf.vectorized_map(lambda x: pid_loss(x[0],x[1],x[2],1,tf.constant(0.5,tf.float32),140,pred),tf.cast(y,tf.float32))
    z = tf.TensorArray(tf.float32,size=100)
    for j in range(p.shape[0]):
        z = z.write(j,cost(tf.stack((P[j],tf.tile(tf.constant([i],tf.float32),P[j].shape),D[j]),1)))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(P,D,z.stack())

def test():
    p = Patient(36,160,60,Gender.F)
    #n = Pharmacodynamic(p,2.0321,2.3266,13.9395,26.6474) 
    n = NNET(p)
    #pid_loss_plot(n,0.0)
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
    #optimized_values =get_PID(n,0.5,swarm_PID(n,0.5))
    #print(time.time()-st)
    #optimized_values =[-10,np.inf,0]
    #c = PID(n,*optimized_values)
    #c = PID(n,-3,-0.003,3)
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
            #bis.extend(b+(int(a>len(refs[int(len(refs)/2):])/2))*0.2+np.random.normal(scale=0.05))
            bis.append(b)
            rs.extend([r])
    plt.figure()
    plt.plot(bis[50:])
    print(time.time()-at)
    #print(c.prop.stack())
