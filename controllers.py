import tensorflow as tf
import data_generator
import math
import time
import scipy
import tensorflow_probability as tfp
import sys
import statsmodels.api as sm
import matplotlib.pyplot as plt
import keras
from keras.models import Model, Sequential
import pso

from anesthesia_models import *
from patient import Patient,Gender


@tf.function
def sequential_append(a,p):
    b = tf.reshape(a,(1,180))[0,:]
    return tf.map_fn(lambda i: tf.reshape(tf.concat((tf.gather(tf.roll(b,-tf.cast(i,tf.int64),axis=0),tf.range(b.shape[0]-i,0,delta=-1,dtype=tf.int64)),tf.gather(p,tf.range(i,dtype=tf.int64))),0),(a.shape[1],1)),tf.range(1,p.shape[0]+1,dtype=a.dtype))

class NNET:
    def __init__(self,p):
        self.z = p.z
        self.saved_model = tf.saved_model.load('./weights')
        self.mdl = tf.function(self.saved_model.signatures['serving_default'])
        self.patient = p
        self.p = 0
        self.r = 0

    @tf.function
    def __call__(self,inputs):
        return self.mdl(input_1=inputs[2],input_2=inputs[1],input_3=inputs[0])['dense_1']

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

class PID(tf.Module):
    def __init__(self,mdl,p=1,i=1,d=1,rho=1):
        self.rho =rho 
        self.pred = mdl
        self.k = tf.Variable(tf.cast([p,i,d],dtype='float32'))
        self.err = tf.Variable(0.5-0.98)
        self.err_i = tf.Variable(0.0)
        self.prop = tf.Variable(tf.zeros((1,180,1)))
        self.remi = tf.Variable(tf.zeros((1,180,1)))
        self.z = tf.Variable(mdl.patient.z)
    
    @tf.function
    def update(self,ref,y):
        u = tf.clip_by_value(stateless_pid(self.k,self.err,self.err_i,y,ref),0,40)
        self.prop.assign(tf.concat((self.prop[:,1:,:],[[[u]]]),axis=1))
        self.remi.assign(self.prop*self.rho)
        self.err_i.assign_add(((ref-y)+self.err)/2)
        self.err.assign(ref-y)
        return self.pred((self.prop,self.prop*self.rho,self.pred.patient.z))[-1][-1]

    @tf.function
    def __call__(self,ref,y):
        return self.update(ref,y)

class AdaptativePID(PID):
    def __init__(self,mdl,p=1,i=1,d=1,rho=1):
        super().__init__(mdl,p=p,i=i,d=d,rho=rho)
        self.iterations = tf.Variable(0)
        self.last_ref = tf.Variable(0.5)
        self.needs_reiter = tf.Variable(False)
    
    @tf.function
    def update(self,ref,y):
        self.iterations.assign_add(1)
        curr_cost = adaptative_cost(tf.expand_dims(self.k,0),y0=y,ref=ref,i0=self.err_i,z=self.pred.z,pred=self.pred,prop=self.prop,remi=self.remi)
        if ref != self.last_ref or self.needs_reiter or curr_cost>50:
            cmp_k = tf.stack([self.k,adapt_PID(self.pred, ref, self.z, y, self.err_i, self.prop, self.remi, x0=self.k)])
            switch_criteria = adaptative_cost(cmp_k,y0=y,ref=ref,i0=self.err_i,z=self.pred.z,pred=self.pred,prop=self.prop,remi=self.remi)
            self.k.assign(cmp_k[tf.math.argmin(switch_criteria)])

            if ref != self.last_ref:
                self.needs_reiter.assign(tf.math.reduce_all(self.k == cmp_k[0]))
            else:
                self.needs_reiter.assign(False)

            if not self.needs_reiter:
                tf.print(self.iterations,switch_criteria,cmp_k[0],self.k)

        self.last_ref.assign(ref)
        return super().update(ref,y)


class MPC(tf.Module):
    def __init__(self,patient: Patient,nnet: Model,horizon=10):
        self.horizon = tf.constant(horizon)
        self.prop = tf.Variable(tf.zeros([1,180,1],dtype=tf.float32))
        self.remi = tf.Variable(tf.zeros([1,180,1],dtype=tf.float32))
        self.pred = nnet
        self.z = self.patient.z
    
    @tf.function
    def update(self,ref,x):
        p,r = self.gen_infusion(ref,x-self.pred((self.prop,self.remi,self.z)))
        self.prop.assign(tf.reshape(tf.concat((tf.transpose(self.prop[0,:,0]),[p[0]]),axis=0)[1:],(1,180,1)))
        self.remi.assign(tf.reshape(tf.concat((tf.transpose(self.remi[0,:,0]),[r[0]]),axis=0)[1:],(1,180,1)))
        return self.pred((self.prop,self.remi,self.z))[-1][-1]
    
    @tf.function
    def gen_infusion(self,ref,bias):
        ref -= bias 
        maxiter = 1e6
        loss = tf.function(lambda x: tfp.math.value_and_gradient(mpc_loss(self.pred,self.patient.z,self.prop,self.remi,self.horizon,x,ref)))
        inputs = tfp.optimize.lbfgs_minimize(loss,initial_position=tf.ones(self.horizon*2))
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
        f = pid_loss(kp,ki,kd,rho,tf.constant(0.5),tf.constant(140),pred)
        grad = tf.gradients(f,[kp,ki,kd])
        tf.print([kp,ki,kd,rho],f,grad)
        return grad 
    return scipy.optimize.minimize(lambda x: tf.cast(pid_loss(*tf.cast(x,tf.float32),tf.constant(1.0),tf.constant(0.5),tf.constant(140),pred),tf.float64),x0,jac=lambda x: tf.cast(gradient(*tf.cast(x,tf.float32),1),tf.float64),method='L-BFGS-B',options={'disp':True}).x

def gd_PID(pred,ref,x0=tf.constant([-1.5,0,0.4])):
    p = tf.Variable(x0[0])
    i = tf.Variable(x0[1])
    d = tf.Variable(x0[2])
    opt = tf.keras.optimizers.Lion(0.01)
    for _ in range(100):
        opt.minimize(lambda: pid_loss(p,i,d,1,0.5,70,pred),[p,i,d])
    return p,i,d

@tf.function
def swarm_PID(pred,ref,z):
    #options = {'c1':0.5, 'c2':0.3, 'w':-0.9}
    #optimizer = ps.single.GlobalBestPSO(n_particles=1100, dimensions=3, options=options)
    return pso.optimize(vectorized_cost,pop_size=300,b=0.7,x_min=-4,x_max=4,dim=3,pred=pred,z=z)

@tf.function
def adapt_PID(pred,ref,z,y,i,prop,remi,x0=None):
    #if not tf.is_tensor(x0):
    return pso.optimize(adaptative_cost,pop_size=50,b=0.7,dim=3,x0=x0,x_min=-1,x_max=1,tol=1,max_iter=48,pred=pred,z=z,y0=y,i0=i,ref=ref,prop=prop,remi=remi)
    #return pso.optimize(adaptative_cost,pop_size=50,b=0.7,dim=3,x0=x0,x_min=-2,x_max=2,tol=adaptative_cost(tf.expand_dims(x0,0),pred=pred,z=z,y0=y,i0=i,ref=ref,prop=prop,remi=remi),max_iter=20,pred=pred,z=z,y0=y,i0=i,ref=ref,prop=prop,remi=remi)

@tf.function
def vectorized_cost(y,pred=None,z=None):
    return pid_loss(y[:,0],y[:,1],y[:,2],1,tf.transpose(tf.repeat(0.5,y.shape[0])),140,pred,tf.repeat(z,y.shape[0],0))

@tf.function
def adaptative_cost(y,pred=None,z=None,y0=0.98,i0=0.0,ref=0.5,prop=None,remi=None):
    return pid_loss(y[:,0],y[:,1],y[:,2],1,tf.transpose(tf.repeat(ref,y.shape[0])),40,pred,tf.repeat(z,y.shape[0],0),y0=y0,i0=i0,prop=prop,remi=remi)

@tf.function
def pid_loss(kp,ki,kd,rho,ref,n,mdl,z,y0=0.98,i0=0.0,prop=None,remi=None):
    last_err = [0.5-0.98]*kp.shape[0]
    i = [i0]*kp.shape[0]
    y = [y0]*kp.shape[0] 
    s = [0.0]*kp.shape[0]

    if not tf.is_tensor(prop):
        prop = tf.zeros((kp.shape[0],180,1))

    if not tf.is_tensor(remi):
        remi = tf.zeros((kp.shape[0],180,1))

    prop = tf.broadcast_to(prop,(kp.shape[0],180,1))
    remi = tf.broadcast_to(remi,(kp.shape[0],180,1))

    for j in range(n):
        err = ref-y
        i += (err+last_err)/2
        pu = tf.clip_by_value(kp*err+ki*i+kd*(err-last_err),0,40)
        ru = tf.clip_by_value(rho*(kp*err+ki*i+kd*(err-last_err)),0,40)

        last_err = err 
        prop = tf.concat((prop[:,1:,:],tf.reshape(pu,(pu.shape[0],1,1))),axis=1)
        remi = tf.concat((remi[:,1:,:],tf.reshape(ru,(ru.shape[0],1,1))),axis=1)
        y = tf.reshape(mdl((prop,remi,z)),kp.shape)
        s += (last_err*10)**2
        if j>30:
            s = tf.where(tf.abs(err)>=0.01,s+50,s) 
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


@tf.function
def run_pid(kp,ki,kd,refs,mdl,z,rho=1):
    j = 0
    kp=tf.cast(kp,tf.float32)
    ki=tf.cast(ki,tf.float32)
    kd=tf.cast(kd,tf.float32)
    last_err = refs[0]-0.98
    i = 0.0
    y = 0.98
    ys = tf.TensorArray(tf.float32,size=refs.shape[0])
    rus = tf.TensorArray(tf.float32,size=refs.shape[0])
    pus = tf.TensorArray(tf.float32,size=refs.shape[0])
    prop = tf.zeros((1,180,1))
    remi = tf.zeros((1,180,1))
    for ref in refs:
        err = ref-y
        i += (err+last_err)/2
        pu = tf.clip_by_value(kp*err+ki*i+kd*(err-last_err),0,40)
        ru = tf.clip_by_value(rho*(kp*err+ki*i+kd*(err-last_err)),0,40)
        last_err = err 
        prop = tf.concat((prop[:,1:,:],[[[pu]]]),axis=1)
        remi = tf.concat((remi[:,1:,:],[[[ru]]]),axis=1)
        y = mdl((prop,remi,z))[-1][-1]
        ys = ys.write(j,y)
        pus = pus.write(j,pu)
        rus = rus.write(j,ru)
        j += 1
    return tf.stack((ys.stack(),pus.stack(),rus.stack()))

@tf.function
def run_controller(c,rfs):
    bis = tf.TensorArray(tf.float32,size=0,dynamic_size=True,clear_after_read=False)
    i = 0
    bis = bis.write(i,c.pred((c.prop,c.prop*c.rho,c.z))[-1][-1])
    for r in rfs:
        b = c.update(r,bis.read(i))
        i += 1
        bis = bis.write(i,b)
        tf.print(i,bis.read(i),r)
    return bis.stack()

def test():
    p = Patient(36,160,60,Gender.F)
    #n = Pharmacodynamic(p,2.0321,2.3266,13.9395,26.6474) 
    n = NNET(p)
    #pid_loss_plot(n,0.0)
    #print(n(tf.zeros((1,180,1)),tf.zeros((1,180,1))))

#    c = MPC(p,n.mdl,5)
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
    optimized_values = swarm_PID(n,0.5,n.z)
    #optimized_values =get_PID(n,0.5)
    #optimized_values = gd_PID(n,0.5)
    #optimized_values =get_PID(n,0.5,swarm_PID(n,0.5))
    #optimized_values =[-10,np.inf,0]
    c = AdaptativePID(n,*optimized_values)
    baseline = PID(n,*optimized_values)
    #c = PID(n,-3,-0.003,3)
#    print(bis)
    refs = data_generator.rfs 
    plt.figure()
    plt.plot(run_controller(c,refs),label='Adaptative')
    plt.plot(run_controller(baseline,refs),label='Fixed parameters')
    plt.plot(refs,label='Reference')
    plt.legend()
    print(time.time()-st,c.k)
    """
    plt.figure()
    plt.title("PSO")
    at = time.time() 
    k = swarm_PID(tf.function(n.mdl),0.5,n.z)
    data = run_pid(k[0],k[1],k[2],refs,n.mdl,p.z)
    plt.plot(data[0])
    plt.plot(refs)
    plt.figure()
    plt.plot(data[1])
    print(time.time()-at)
    """
    """
    plt.figure()
    plt.title("L-BFGS")
    at = time.time() 
    plt.plot(run_controller(PID(n,*get_PID(n,0.5)),0.5,600))
    print(time.time()-at)
    plt.plot([0.5]*600)
    plt.figure()
    plt.title("PSO+L-BFGS")
    at = time.time() 
    plt.plot(run_controller(PID(n,*get_PID(n,0.5,swarm_PID(n,0.5))),0.5,600))
    print(time.time()-at)
    plt.plot([0.5]*600)
    #print(c.prop.stack())
    """
