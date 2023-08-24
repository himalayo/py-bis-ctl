import matplotlib.pyplot as plt
import numpy as np
import math
import control as ct
import time
from patient import Patient,Gender
from keras.models import load_model

mean_c =np.array([ 56.50928621,0.49630326,61.13655517,162.0409402])
std_c =np.array([15.38680592, 0.49998633, 9.23908289, 8.33116551])

class Predictor:
    def __call__(self,u: np.ndarray):
        return ct.input_output_response(self.io,np.arange(u.shape[1]),u,X0=self.x0).outputs

class PKPD(Predictor):
    def __init__(self,beta,gamma,cp50,cr50):
        def hill(beta,gamma,cp50,cr50,prop,remi):
            u_prop = prop/cp50
            u_remi = remi/cr50
            phi = (u_prop)/(u_prop+u_remi)
            u50 = 1 - (beta*phi) + (beta*phi**2)
            return 97.7  * ( 1- ( ( (( u_prop + u_remi )/u50 )**gamma )/( 1 + ( ((u_prop + u_remi)/u50)**gamma) ) ) )
        self.io = ct.NonlinearIOSystem(lambda t,x,u,params: x,lambda t,x,u,params=None: hill(beta,gamma,cp50,cr50,u[0],u[1]),inputs=2,states=2,outputs=1)
        self.x0=[0,0]

class Neural(Predictor):
    def __init__(self,path,patient: Patient):
        self.x0 = 0.97
        def state_update(t,x,u,params):
            st = time.time()  
            def accum(seq):
                out = np.zeros(180)
                np.copyto(out[-min(math.ceil(len(seq)/10),180):],list(map(sum,np.array_split(seq,math.ceil(len(seq)/10))[-180:])))
                return out

            if t <= 1e-4:
                params['prop'] = [u[0]]
                params['remi'] = [u[1]]
                params['prop_input'] = accum(np.array([u[0]]))
                params['remi_input'] = accum(np.array([u[1]]))
                params['micro_prop'] = [u[0]]
                params['micro_remi'] = [u[1]]
                params['t'] = [t]
                params['last_t'] = t
                params['nnet'].reset_states()
            elif t-params['last_t'] >= 1:
                params['micro_prop'].append(u[0])
                params['micro_remi'].append(u[1])
                params['prop'].append(sum(params['micro_prop']))
                params['remi'].append(sum(params['micro_remi']))
                np.append(params['prop_input'],accum(np.array(params['prop'])))
                np.append(params['remi_input'],accum(np.array(params['remi'])))
                params['t'].append(t)
                params['last_t'] = t
                params['micro_prop'] = []
                params['micro_remi'] = []
            else:
                params['micro_prop'].append(u[0])
                params['micro_remi'].append(u[1])
            """
            def gen_nnet_inputs(remi,prop,patient):
                def to_case(remi,prop,patient):
                    clinical = np.broadcast_to((patient.np-mean_c)/std_c,[remi.shape[0],4])
                    return np.hstack((remi,prop,clinical))
                case = to_case(remi,prop,patient)
                

                timepoints = 180
                # make sequence
                gaps = np.arange(0, 10 * (timepoints + 1), 10)
                ppf_seq = []
                rft_seq = []

                case_p= []
                case_r = []
                case_c = []
                for i,row in enumerate(case):
                    ppf_dose = row[0]
                    rft_dose = row[1]
                    age = row[2]
                    sex = row[3]
                    wt = row[4]
                    ht = row[5]
                    ppf_seq.append(ppf_dose)  # make time sequence
                    rft_seq.append(rft_dose)

                    pvals = []
                    rvals = []
                    for j in reversed(range(timepoints)):
                        istart = i + 1 - gaps[j + 1]
                        iend = i + 1 - gaps[j]
                        pvals.append(sum(ppf_seq[max(0, istart):max(0, iend)]))
                        rvals.append(sum(rft_seq[max(0, istart):max(0, iend)]))

                
                    case_p.append(pvals)
                    case_r.append(rvals)
                    case_c.append([age, sex, wt, ht])
                case_p = np.array(case_p)
                case_r = np.array(case_r)
                return case_p.reshape(case_p.shape[0],case_p.shape[1],1)/12, case_r.reshape(case_r.shape[0],case_r.shape[1],1)/12, np.array(case_c)
            """
            out = params['nnet'].predict((params['prop_input'].reshape((math.ceil(params['prop_input'].shape[0]/180),180,1)),params['remi_input'].reshape((math.ceil(params['remi_input'].shape[0]/180),180,1)),np.broadcast_to((params['patient'].np-mean_c)/std_c,[math.ceil(params['remi_input'].shape[0]/180),4])),verbose=None)
            return out[-1]-x 

        self.nnet = load_model(path)
        self.params = {'prop': [], 'remi': [], 'patient': patient,'nnet':self.nnet}
        self.io = ct.NonlinearIOSystem(state_update,lambda t,x,u,params: x,params=self.params,inputs=2,states=1,outputs=1)

def test():
   plt.figure()
   plt.plot(PKPD(1,1.5,5,20)(np.array([np.sin(np.linspace(0,2*np.pi,100))+1,np.sin(np.linspace(0,3*np.pi,100))+1]))[0])
   plt.figure()
   plt.plot(Neural('./weights',Patient(50,170,70,Gender.F))(np.array([np.sin(np.linspace(0,2*np.pi,50))*1e-5+1e-5,np.sin(np.linspace(0,3*np.pi,50))*1e-5+1e-5]))[0])
   plt.figure()
   plt.plot(Neural('./weights',Patient(50,170,70,Gender.F))(np.ones([2,50]))[0])
