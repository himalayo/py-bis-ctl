import os
import tensorflow as tf
#import jax
import data_generator
import math
import time
import sys
#import statsmodels.api as sm
import matplotlib.pyplot as plt
import tf_keras as keras
import scipy
from keras.models import Model, Sequential
import pso
import control as ct
#from anesthesia_models import *
from patient import Patient,Gender



@tf.function
def sequential_append(a,p):
    b = tf.reshape(a,(1,180))[0,:]
    return tf.map_fn(lambda i: tf.reshape(tf.concat((tf.gather(tf.roll(b,-tf.cast(i,tf.int64),axis=0),tf.range(b.shape[0]-i,0,delta=-1,dtype=tf.int64)),tf.gather(p,tf.range(i,dtype=tf.int64))),0),(a.shape[1],1)),tf.range(1,p.shape[0]+1,dtype=a.dtype))

class NNET:
    def __init__(self,p):
        self.z = p.z
        self.mdl = tf.function(keras.models.load_model('./weights'))
        self.patient = p

    @tf.function
    def __call__(self,x):
        return self.mdl(x)

class StateSpace(tf.Module):
    def __init__(self,A,B,C):
        self.A = tf.cast(A,tf.float32)
        self.B = tf.cast(B,tf.float32)
        self.C = tf.cast(C,tf.float32)
        self.x = tf.Variable(tf.zeros((self.A.shape[0],1)))
        self.vector_A = tf.reverse(tf.scan(lambda a,x: a@x, tf.tile(tf.expand_dims(tf.linalg.expm(self.A*10),0),[180,1,1])),[0])
        self.vector_B = tf.tile(tf.expand_dims(tf.linalg.inv(self.A)@(tf.linalg.expm(self.A*10)-tf.eye(self.x.shape[0]))@self.B,0),[180,1,1])


    @staticmethod
    def from_pk(v,k):
        return StateSpace(tf.constant([[-(k[0]+k[1]+k[2]),(v[1]/v[0])*k[3], (v[2]/v[0])*k[4], 0],
                                       [(v[0]/v[1])*k[1], -k[3], 0, 0],
                                       [(v[0]/v[2])*k[2],0,-k[4],0],
                                       [k[4], 0, 0, -k[4]]
                                        ]),
                          tf.constant([[1/v[0]], [0], [0], [0]]),
                          tf.constant([[0, 0, 0, 1]]))
    
    @tf.function
    def vectorized(self,u):
        return self.C@tf.reduce_sum(self.vector_A@(self.vector_B@tf.reshape(tf.cast(u,tf.float32),(u.shape[0],180,1,1))),axis=1)

    def __iter__(self):
        yield self.A
        yield self.B
        yield self.C

    def stateless(self,x,u,dt=tf.constant(10)):
        bu = (self.B*u)
        new_x = tf.cast(tf.linalg.expm(self.A*dt)@x + tf.linalg.inv(self.A)@(tf.linalg.expm(self.A*dt)-tf.eye(x.shape[0]))@bu,tf.float32)
        return new_x
    @tf.function
    def __call__(self,u,dt=tf.constant(10,dtype=tf.float32)):
        self.x.assign(tf.linalg.expm(self.A*dt)@self.x + tf.linalg.inv(self.A)@(tf.linalg.expm(self.A*dt)-tf.eye(self.x.shape[0]))@self.B*u)
        return self.C@self.x

class Schnider(StateSpace):
    def __init__(self,patient):
        v = [
            4.27,
            18.9 - 0.391 * (patient.age - 53),
            238.0]

        k = [
            0.433 + 0.0107*(patient.weight-77) - 0.00159*(patient.lbm - 59) + 0.0062*(patient.height-177),
            0.302 - 0.0056 * (patient.age - 53),
            0.196,
            (1.29 - 0.024*(patient.age-53))/(18.9 - 0.391*(patient.age - 53)),
            0.0035,
            0.456
        ]

        k = [x/60 for x in k]

        super().__init__(*StateSpace.from_pk(v,k))

class Minto(StateSpace):
    def __init__(self,patient):
        v = [
            5.1 - 0.0201*(patient.age - 40) + 0.072*(patient.lbm - 55),
            9.82 - 0.811*(patient.age - 40) + 0.108*(patient.lbm - 55),
            5.42
        ]

        k = [
            (2.6 - 0.0162*(patient.age-40) + 0.0191*(patient.lbm-55))/v[0],
             (2.05 - 0.0301 * (patient.age-40))/v[0],
             (0.076 - 0.00113*(patient.age-40))/v[0],
             (2.05 - 0.0301 * (patient.age-40))/v[1],
             (0.076 - 0.00113*(patient.age-40))/v[2],
             0.595 - 0.007*(patient.age-40)
        ]

        k = [x/60 for x in k]
        super().__init__(*StateSpace.from_pk(v,k))

    
class Pharmacodynamic:
    def __init__(self,patient,beta,gamma,cp50,cr50):
        self.patient = patient
        self.beta = beta
        self.gamma = gamma
        self.cp50=cp50
        self.cr50=cr50
        self.z = patient.z
        self.mdl = self.__call__

        self.propofol = Schnider(patient)
        self.remifentanil = Minto(patient)


        if not tf.is_tensor(self.beta):
            self.beta = tf.constant(self.beta,dtype=tf.float32)

        if not tf.is_tensor(self.cp50):
            self.cp50 = tf.constant(self.cp50,dtype=tf.float32)

        if not tf.is_tensor(self.cr50):
            self.cr50 = tf.constant(self.cr50,dtype=tf.float32)

        if not tf.is_tensor(self.gamma):
            self.gamma = tf.constant(self.gamma,dtype=tf.float32)

    @tf.function
    def hill(self,prop,remi):
        u_prop = tf.math.xdivy(prop,self.cp50)
        u_remi = tf.math.xdivy(remi,self.cr50)

        phi = tf.math.xdivy(u_prop,(u_prop+u_remi))

        u50 = 1 - (self.beta*phi) + (self.beta*phi**2)
        r= ( 1- ( ( (( u_prop + u_remi )/u50 )**self.gamma )/( 1 + ( ((u_prop + u_remi)/u50)**self.gamma) ) ) )
        print(u_prop,u_remi,phi,u50,r)
        return tf.cast(r,tf.float32)

    @tf.function
    def linear(self,p,r,dt=tf.constant(10)):
        #return tf.stack(tf.scan(self.propofol.stateless,p,tf.zeros((4,1)))[:,-1,0],tf.scan(self.remifentanil.stateless,r,tf.zeros((4,1)))[:,-1,0]))
        return tf.stack((self.propofol.vectorized(p),self.remifentanil.vectorized(r)))

    @tf.function
    def __call__(self,vals):
        #return tf.map_fn(lambda x: self.hill(*tf.unstack(self.linear(x[0][:,0],x[1][:,0])))[-1], (vals[0],vals[1]),fn_output_signature=tf.float32)
        return self.hill(*tf.unstack(self.linear(vals[0],vals[1])))


class Controller(tf.Module):
    def __init__(self,pred,name=None):
        super().__init__(name)
        self.pred = pred
        self.z = pred.patient.z
        self.patient = pred.patient
        super().__setattr__('prop',tf.Variable(tf.zeros((1,180,1))))
        super().__setattr__('remi',tf.Variable(tf.zeros((1,180,1))))
        self.type = tf.strings.as_string('base')
    
    def __setattr__(self,attribute,value):
        if attribute=='prop':
            self.prop.assign(value)
            return

        if attribute=='remi':
            self.remi.assign(value)
            return
        
        super().__setattr__(attribute,value)
    
    @tf.function
    def set_infusion(self, prop, remi):
        self.prop.assign(prop)
        self.remi.assign(remi)

    @tf.function
    def applied_infusion(self,p,r):
        return tf.reshape(tf.concat((self.prop[:,1:,:],[[[p]]]),axis=1),(1,180,1)), tf.reshape(tf.concat((self.remi[:,1:,:],[[[r]]]),axis=1),(1,180,1))

    @tf.function
    def bias(self,y):
        return y-self.pred((self.prop,self.remi,self.z))
    
    @tf.function
    def update(self,ref,y,display=True):
        p,r = self.gen_infusion(ref,y)
        prop,remi = self.applied_infusion(p,r)
        self.set_infusion(prop,remi)
        out = self.status()

        if display:
            tf.print(self.type,ref,out)

        self.update_callback(ref,y)
        return out 
    
    @tf.function
    def sync(self,other):
        self.prop = other.prop
        self.remi = other.remi

    @tf.function
    def status(self):
        return tf.squeeze(self.pred((self.prop,self.remi,self.z)))

    @tf.function
    def __call__(self,ref,y,display=True):
        return self.update(ref,y,display=display)

    @tf.function
    def predict(self,p,r):
        return self.pred((*self.applied_infusion(p,r),self.z))
    
    @tf.function
    def update_callback(self,ref,y):
        pass

    @tf.function
    def open(self,ref):
        return self.update(ref,self.status())

@tf.function
def stateless_pid(k,last_u,other_e,last_e,y,ref,dt=tf.constant(10.0)):
    u = last_u + \
        ( (k[0] + (k[1]*dt) + (k[2]/dt) ) * (ref-y) ) + \
        (( -k[0] -( (2 * k[2]) / dt ) ) * last_e)+ \
        ( (k[2] / dt) * other_e )
    return u


class dPID(Controller):
    def __init__(self,mdl,p=1,i=1,d=1,rp=1,ri=1,rd=1,k=None,rk=None):
        super().__init__(mdl)
        self.type = tf.strings.as_string('pid')
        if k is None:
            self.pk = tf.Variable(tf.cast([p,i,d],dtype='float32'))
        else:
            self.pk = tf.identity(pk)
        if rk is None:
            self.rk = tf.Variable(tf.cast([rp,ri,rd],dtype='float32'))
        else:
            self.rk = tf.identity(pk)


        self.err = tf.Variable(0.5-0.98)
        self.last_err = tf.Variable(tf.identity(self.err))

    @tf.function
    def gen_infusion(self,ref,y):
        ref -= self.bias(y)
        pu = tf.squeeze(tf.clip_by_value(
            stateless_pid(self.pk,self.prop[0,-1,0],self.last_err,self.err,y,ref)
            ,0,40))
        ru = tf.squeeze(tf.clip_by_value(
            stateless_pid(self.rk,self.remi[0,-1,0],self.last_err,self.err,y,ref)
            ,0,40))
        return pu,ru
    
    @tf.function
    def update_callback(self,ref,y):
        self.last_err.assign(tf.identity(self.err))
        self.err.assign(ref-y)

    def identity(self):
        copy = PID(self.pred,*tf.unstack(tf.identity(self.pk)),tf.identity(self.rk))
        copy.set_infusion(tf.identity(self.prop),tf.identity(self.remi))
        copy.err.assign(tf.identity(self.err))
        copy.last_err.assign(tf.identity(self.last_err))
        return copy



class PID(Controller):
    def __init__(self,mdl,p=1,i=1,d=1,rho=1,k=None):
        super().__init__(mdl)
        self.type = tf.strings.as_string('pid')
        self.rho =tf.cast(rho,tf.float32)
        if k is None:
            self.k = tf.Variable(tf.cast([p,i,d],dtype='float32'))
        else:
            self.k = tf.identity(k)

        self.err = tf.Variable(0.5-0.98)
        self.last_err = tf.Variable(tf.identity(self.err))

    @tf.function
    def gen_infusion(self,ref,y):
        ref -= self.bias(y)
        u = tf.squeeze(tf.clip_by_value(
            stateless_pid(self.k,self.prop[0,-1,0],self.last_err,self.err,y,ref)
            ,0,40))
        return u,tf.clip_by_value(u*self.rho,0,40)
    
    @tf.function
    def update_callback(self,ref,y):
        self.last_err.assign(tf.identity(self.err))
        self.err.assign(ref-y)

    def identity(self):
        copy = PID(self.pred,*tf.unstack(tf.identity(self.k)),tf.identity(self.rho))
        copy.set_infusion(tf.identity(self.prop),tf.identity(self.remi))
        copy.err.assign(tf.identity(self.err))
        copy.last_err.assign(tf.identity(self.last_err))
        return copy


class AdaptativePID(PID):
    def __init__(self,mdl,p=1,i=1,d=1,rho=1,k=None):
        super().__init__(mdl,p=p,i=i,d=d,rho=rho,k=k)
        self.type = tf.strings.as_string('adaptative pid')
        self.iterations = tf.Variable(0)
        self.last_ref = tf.Variable(0.5)
        self.needs_reiter = tf.Variable(False)

    
    @tf.function
    def update(self,ref,y,display=True):
        self.iterations.assign_add(1)
        curr_cost = adaptative_cost(tf.reshape([self.k[1]],(1,1)),y0=y,ref=ref,i0=self.err_i,z=self.pred.z,pred=self.pred,prop=self.prop,remi=self.remi,p0=tf.reshape((self.k[0],),(1,)),d0=tf.reshape((self.k[2],),(1,)))
        if ref != self.last_ref or self.needs_reiter or curr_cost>10:
            cmp_k = tf.stack([self.k,adapt_PID(self.pred, ref, self.z, y, self.err_i, self.prop, self.remi,self.k[0],self.k[2], x0=self.k[1])])
            switch_criteria = adaptative_cost(cmp_k,y0=y,ref=ref,i0=self.err_i,z=self.pred.z,pred=self.pred,prop=self.prop,remi=self.remi,p0=self.k[0],d0=self.k[2])
            self.k.assign(cmp_k[tf.math.argmin(switch_criteria)])

            if ref != self.last_ref:
                self.needs_reiter.assign(tf.math.reduce_all(self.k == cmp_k[0]))
            else:
                self.needs_reiter.assign(False)

            if not self.needs_reiter:
                tf.print(self.iterations,switch_criteria,cmp_k[0],self.k)

        self.last_ref.assign(ref)
        return super().update(ref,y,display=display)

class MPC(Controller):
    def __init__(self,patient: Patient,nnet: Model,horizon=5,internal_k=tf.constant([-3,-0.003,3])):
        with tf.device('GPU:0'):
            super().__init__(nnet,name="lmao")
            self.type = tf.strings.as_string('mpc')
            self.horizon = horizon
            self.intermediate = tf.Variable(0.0)
            self.pid = PID(nnet,k=internal_k)

    @tf.function
    def _py_gen_infusion(self,ref,y):
        loss = lambda x,r: mpc_loss(self.pred,self.patient.z,self.prop,self.remi,self.horizon,x,r)
        j = lambda x,r: mpc_loss_jac(self.pred,self.patient.z,self.prop,self.remi,self.horizon,x,r)

        def opt_fn(x0,r):
            start = time.perf_counter_ns()
            out = scipy.optimize.minimize(loss,x0,jac=j,method='L-BFGS-B',bounds=[(0.0,30)]*int(x0.shape[0]),args=(r,),options={'disp':False}).x
            print((time.perf_counter_ns()-start)*1e-9)
            return out

        #opt_fn = lambda x0,r: scipy.optimize.minimize(loss,x0,jac=j,method='L-BFGS-B',args=(r,),options={'disp':False}).x
        p,r = self.pid.gen_infusion(ref,y)
        inputs = tf.squeeze(tf.py_function(opt_fn,inp=[tf.ones(self.horizon*2)*p,ref],Tout=[tf.float32]))
        return inputs[:self.horizon][0],inputs[self.horizon:][0]


    @tf.function
    def update_callback(self,ref,y):
        self.pid.set_infusion(tf.identity(self.prop),tf.identity(self.remi))
        self.pid.update_callback(ref,y)

    @tf.function
    def gen_infusion(self,ref,y):
        return self._py_gen_infusion(ref,y)

    def identity(self):
        copy = MPC(self.pred.patient,self.pred,self.horizon)
        copy.set_infusion(tf.identity(self.prop),tf.identity(self.remi))
        return copy



class Mixed(Controller):
    def __init__(self,predictor, threshold=0.03, mpc=None, pid=None):
        super().__init__(predictor)
        self.type = tf.strings.as_string('mixed')

        if pid == None:
            pid = PID(self.pred, *swarm_PID(self.pred,0.5,self.pred.z))

        if mpc == None:
            mpc = MPC(self.pred.patient, self.pred,horizon=10,internal_k=pid.k)
        

        self.mpc = mpc
        self.pid = pid

    @tf.function
    def update(self,ref,y,display=True):

        mpc_infusion = tf.stack(self.mpc.applied_infusion(*self.mpc.gen_infusion(ref,y)),0)
        pid_infusion = tf.stack(self.pid.applied_infusion(*self.pid.gen_infusion(ref,y)),0)
        mpc_prop, mpc_remi = tf.unstack(mpc_infusion)
        pid_prop, pid_remi = tf.unstack(pid_infusion)
        mpc_y = self.pred((mpc_prop, mpc_remi, tf.identity(self.mpc.z)))
        pid_y = self.pred((pid_prop, pid_remi, tf.identity(self.pid.z)))


        if tf.math.abs(ref-mpc_y) < tf.math.abs(ref-pid_y):
            prop = mpc_prop
            remi = mpc_remi
            tf.print(self.type,self.mpc.type,ref, mpc_y)
        else:
            prop = pid_prop
            remi = pid_remi
            tf.print(self.type,self.pid.type,ref, pid_y)

        self.set_infusion(prop, remi)
        self.pid.set_infusion(prop,remi)
        self.mpc.set_infusion(prop,remi)
        self.pid.update_callback(ref,y)
        self.mpc.update_callback(ref,y)
        return self.status()

    @tf.function
    def comparison(self,controller_1, controller_2, refs):
        return tf.parallel_stack((tf.map_fn(self.open,refs,parallel_iterations=50),tf.map_fn(controller_1.open,refs,parallel_iterations=50),tf.map_fn(controller_2.open,refs,parallel_iterations=50)))

    def compare(self, refs):
        mpc_copy = self.mpc.identity()
        pid_copy = self.pid.identity()
        return self.comparison(mpc_copy,pid_copy,tf.identity(refs))

    def identity(self):
        copy = Mixed(self.pred,mpc=self.mpc.identity(),pid=self.pid.identity())
        copy.set_infusion(tf.identity(self.prop),tf.identity(self.remi))
        return copy

@tf.function
def cond(converged, failed):
    tf.print(converged, failed)
    return tfp.optimizer.converged_any(converged,failed)

@tf.function
def mpc_loss_jac(nnet,z,prop,remi,horizon,x,ref):
    x = tf.cast(x,prop.dtype)
    a = 0.0
    for i in range(1,horizon):
        err = tf.math.abs(ref-nnet((tf.reshape(tf.concat((prop[0,:,0],x[:i+1]),axis=0)[-180:],(1,180,1)),tf.reshape(tf.concat((remi[0,:,0],x[horizon:horizon+i+1]),axis=0)[-180:],(1,180,1)),z))[-1][-1])
        a += ((err*((1e8+1.0) if err >= 0.05 else 1.0))**2)/i
    tf.print(tf.gradients(a,x))
    return tf.gradients(a,x)[0]

@tf.function
def mpc_loss(nnet,z,prop,remi,horizon,x,ref):
    x = tf.cast(x,prop.dtype)
    a = 0.0
    for i in range(1,horizon):
        err = tf.math.abs(ref-nnet((tf.reshape(tf.concat((prop[0,:,0],x[:i+1]),axis=0)[-180:],(1,180,1)),tf.reshape(tf.concat((remi[0,:,0],x[horizon:horizon+i+1]),axis=0)[-180:],(1,180,1)),z))[-1][-1])
        a += ((err*((1e8+1.0) if err >= 0.05 else 1.0))**2)/i

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

def get_dPID(pred,ref,x0=tf.constant([-6,-0.005,10,-6,-0.005,10])):
    @tf.function
    def gradient(kp,ki,kd,rkp,rki,rkd):
        f = double_pid_loss(kp,ki,kd,rkp,rki,rkd,tf.constant(0.5),tf.constant(250),pred,0)
        grad = tf.gradients(f,[kp,ki,kd,rkp,rki,rkd])
        tf.print([kp,ki,kd,rkp,rki,rkd],f,grad)
        return grad 
    return scipy.optimize.minimize(lambda x: tf.cast(double_pid_loss(*tf.cast(tf.expand_dims(x,-1),tf.float32),tf.constant(0.5),tf.constant(250),pred,0),tf.float64),x0,jac=lambda x: tf.cast(gradient(*tf.cast(tf.expand_dims(x,-1),tf.float32)),tf.float64),method='L-BFGS-B',options={'disp':True}).x


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
    return pso.optimize(vectorized_cost,pop_size=500,tol=1e+16,b=0.7,x_min=-5,x_max=5,dim=4,pred=pred,z=z)

@tf.function
def double_swarm_PID(pred,ref,z):
    #options = {'c1':0.5, 'c2':0.3, 'w':-0.9}
    #optimizer = ps.single.GlobalBestPSO(n_particles=1100, dimensions=3, options=options)
    return pso.optimize(double_vectorized_cost,pop_size=7500,tol=1e+16,b=0.7,x_min=-5,x_max=5,dim=6,pred=pred,z=z)


@tf.function
def adapt_PID(pred,ref,z,y,i,prop,remi,p0,d0,x0=None):
    #if not tf.is_tensor(x0):
    return pso.optimize(adaptative_cost,pop_size=50,b=0.7,dim=1,x0=x0,x_min=-1,x_max=1,tol=1,max_iter=20,pred=pred,z=z,y0=y,i0=i,ref=ref,prop=prop,remi=remi,p0=tf.broadcast_to((p0,),(50,1)),d0=tf.broadcast_to((d0,),(50,1)))
    #return pso.optimize(adaptative_cost,pop_size=50,b=0.7,dim=3,x0=x0,x_min=-2,x_max=2,tol=adaptative_cost(tf.expand_dims(x0,0),pred=pred,z=z,y0=y,i0=i,ref=ref,prop=prop,remi=remi),max_iter=20,pred=pred,z=z,y0=y,i0=i,ref=ref,prop=prop,remi=remi)

@tf.function
def vectorized_cost(y,pred=None,z=None):
    return pid_loss(y[:,0],y[:,1],y[:,2],y[:,3],tf.transpose(tf.repeat(0.5,y.shape[0])),500,pred,tf.repeat(z,y.shape[0],0))

@tf.function
def double_vectorized_cost(y,pred=None,z=None):
    return double_pid_loss(y[:,0],y[:,1],y[:,2],y[:,3],y[:,4],y[:,5],tf.transpose(tf.repeat(0.5,y.shape[0])),300,pred,tf.repeat(z,y.shape[0],0))


@tf.function
def adaptative_cost(y,pred=None,z=None,y0=0.98,i0=0.0,ref=0.5,prop=None,remi=None,p0=None,d0=None):
    return pid_loss(p0,y[:,0],d0,1,tf.transpose(tf.repeat(ref,y.shape[0])),30,pred,tf.repeat(z,y.shape[0],0),y0=y0,i0=i0,prop=prop,remi=remi)

@tf.function
def double_pid_loss(kp,ki,kd,rkp,rki,rkd,ref,n,mdl,z,y0=0.98,i0=0.0,prop=None,remi=None):
    last_err = tf.fill(kp.shape,1-ref)
    i = tf.fill(kp.shape,i0)
    y = tf.fill(kp.shape,y0) 
    s = tf.zeros(kp.shape)
    punish = tf.fill(kp.shape,1e8+1.0)
    reward = tf.fill(kp.shape,1.0)
    other_err = tf.identity(last_err)
    print(rkd)
    pk = tf.concat([tf.expand_dims(kp,0),tf.expand_dims(ki,0),tf.expand_dims(kd,0)],0)
    rk = tf.concat([tf.expand_dims(rkp,0),tf.expand_dims(rki,0),tf.expand_dims(rkd,0)],0)

    if not tf.is_tensor(prop):
        prop = tf.zeros((kp.shape[0],180,1))

    if not tf.is_tensor(remi):
        remi = tf.zeros((kp.shape[0],180,1))

    prop = tf.broadcast_to(prop,(kp.shape[0],180,1))
    remi = tf.broadcast_to(remi,(kp.shape[0],180,1))

    for j in range(n):
        pu = tf.clip_by_value(stateless_pid(pk,prop[:,-1,0],other_err,last_err,y,ref),0,40)
        ru = tf.clip_by_value(stateless_pid(rk,remi[:,-1,0],other_err,last_err,y,ref),0,40)

        other_err = tf.identity(last_err)
        last_err = ref-y
        prop = tf.concat((prop[:,1:,:],tf.reshape(pu,(pu.shape[0],1,1))),axis=1)
        remi = tf.concat((remi[:,1:,:],tf.reshape(ru,(ru.shape[0],1,1))),axis=1)
        y = tf.reshape(mdl((prop,remi,z)),kp.shape)
        err = ref-y
        if j>60:
            s += ((err*tf.where(tf.math.abs(err)<=0.05,reward,punish))**2)
            continue
        #tf.print(tf.math.reduce_min(tf.math.abs(err)),rho[tf.math.argmin(tf.abs(err))])
        s += err**2
    tf.print(prop[tf.math.argmin(s)],remi[tf.math.argmin(s)])
    return s


@tf.function
def pid_loss(kp,ki,kd,rho,ref,n,mdl,z,y0=0.98,i0=0.0,prop=None,remi=None):
    last_err = [0.5-0.98]*kp.shape[0]
    i = [i0]*kp.shape[0]
    y = [y0]*kp.shape[0] 
    s = [0.0]*kp.shape[0]
    punish = tf.fill(kp.shape,1e8+1.0)
    reward = tf.fill(kp.shape,1.0)
    other_err = tf.identity(last_err)
    k = tf.concat([tf.expand_dims(kp,0),tf.expand_dims(ki,0),tf.expand_dims(kd,0)],0)

    if not tf.is_tensor(prop):
        prop = tf.zeros((kp.shape[0],180,1))

    if not tf.is_tensor(remi):
        remi = tf.zeros((kp.shape[0],180,1))

    prop = tf.broadcast_to(prop,(kp.shape[0],180,1))
    remi = tf.broadcast_to(remi,(kp.shape[0],180,1))

    for j in range(n):
        u = tf.clip_by_value(stateless_pid(k,prop[:,-1,0],other_err,last_err,y,ref),0,40)
        pu = u
        ru = tf.clip_by_value(rho*u,0,40)

        other_err = tf.identity(last_err)
        last_err = ref-y
        prop = tf.concat((prop[:,1:,:],tf.reshape(pu,(pu.shape[0],1,1))),axis=1)
        remi = tf.concat((remi[:,1:,:],tf.reshape(ru,(ru.shape[0],1,1))),axis=1)
        y = tf.reshape(mdl((prop,remi,z)),kp.shape)
        err = ref-y
        if j>60:
            s += ((err*tf.where(tf.math.abs(err)<=0.05,reward,punish))**2)
            continue
        tf.print(tf.math.reduce_min(tf.math.abs(err)),rho[tf.math.argmin(tf.abs(err))])
        s += err**2
    tf.print(prop[tf.math.argmin(s)],remi[tf.math.argmin(s)])
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
    with tf.device('/gpu:0'):
        bis = tf.TensorArray(tf.float32,size=0,dynamic_size=True,clear_after_read=False)
        i = 0
        bis = bis.write(i,c.status())
        for r in rfs:
            b = c.update(r,bis.read(i))
            i += 1
            bis = bis.write(i,b)
            tf.print(i,bis.read(i),r)
    return bis.stack()
def test():
    p = Patient(40,160,60,Gender.F)
    #n = NNET(p)
    n = Pharmacodynamic(p,2.0321,2.3266,13.9395,26.6474) 
    #c = PID(n,-6,-0.005,10) # faster debugging
    #c = MPC(p,n)
    #c = dPID(n,*get_dPID(n,0.5))
    c = dPID(n,-6,-0.005,10,-6,-0.005,10) # faster debugging
    print(double_pid_loss(tf.unstack(tf.expand_dims(tf.constant((-6,-0.005,10,-6,-0.005,10))))),[0.5],250,n,0)
    #print(c.rho)
    refs = tf.ones(1000)*0.5
    print(run_controller(c,refs))
    print(n.propofol.vectorized(c.prop),ct.input_output_response(ct.StateSpace(n.propofol.A,n.propofol.B,n.propofol.C,0).sample(10),tf.range(1,1800,10),tf.reshape(c.prop,180)).outputs[-1])
    """
    #n = NNET(p)
    #pid_loss_plot(n,0.0)
    #print(n(tf.zeros((1,180,1)),tf.zeros((1,180,1))))

    c = Mixed(n)
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
    #optimized_values = swarm_PID(n,0.5,n.z) 
    #optimized_values =get_PID(n,0.5)
    #optimized_values = gd_PID(n,0.5)
    #optimized_values =get_PID(n,0.5,swarm_PID(n,0.5))
    #optimized_values =[-10,np.inf,0]
    #c = AdaptativePID(n,*optimized_values)
    #baseline = PID(n,*optimized_values)
    #baseline = PID(n,-3,-0.003,3)
#    print(bis)
    refs = data_generator.rfs
    #refs = tf.ones(500)*0.5
    mixed, mpc, pid = c.compare(refs)
    print(mixed)
    plt.figure()
    plt.plot(mixed,label='Mixed control')
    plt.plot(mpc,label='MPC')
    plt.plot(pid,label='PID')
    plt.plot(refs,label='Reference')
    plt.legend()
    plt.savefig("test.pdf")
    """
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
