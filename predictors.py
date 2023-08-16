import matplotlib.pyplot as plt
import numpy as np
import control as ct
from patient import Patient,Gender
from keras.models import load_model




class Predictor:
    def __call__(self,u: np.ndarray):
        return ct.input_output_response(self.io,np.arange(u.shape[1]),u).outputs

class PKPD(Predictor):
    def __init__(self,beta,gamma,cp50,cr50):
        def hill(beta,gamma,cp50,cr50,prop,remi):
            u_prop = prop/cp50
            u_remi = remi/cr50
            phi = (u_prop)/(u_prop+u_remi)
            u50 = 1 - (beta*phi) + (beta*phi**2)
            return 97.7  * ( 1- ( ( (( u_prop + u_remi )/u50 )**gamma )/( 1 + ( ((u_prop + u_remi)/u50)**gamma) ) ) )
        self.io = ct.NonlinearIOSystem(lambda t,x,u,params: x,lambda t,x,u,params=None: hill(beta,gamma,cp50,cr50,u[0],u[1]),inputs=2,states=2,outputs=1)

class Neural(Predictor):
    def __init__(self,path,patient: Patient):
        def state_update(t,x,u,params):
            params['prop'].append(u[0])
            params['remi'].append(u[1])
            prop = np.array(params['prop']).reshape([len(params['prop']),1])
            remi = np.array(params['remi']).reshape([len(params['remi']),1])
            def gen_nnet_inputs(remi,prop,patient):
                def to_case(remi,prop,patient):
                    clinical = np.broadcast_to(patient.np,[remi.shape[0],4])
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
            out = params['nnet'](gen_nnet_inputs(prop,remi,params['patient']))
            print(prop[-1],remi[-1],out[-1])
            return out[-1]
        self.nnet = load_model(path)
        self.params = {'prop': [], 'remi': [], 'patient': patient,'nnet':self.nnet}
        self.io = ct.NonlinearIOSystem(lambda t,x,u,params: u,outfcn=state_update,params=self.params,inputs=2,states=2,outputs=1)

def test():
   plt.figure()
   plt.plot(PKPD(1,1.5,5,20)(np.array([np.sin(np.linspace(0,2*np.pi,100))+1,np.sin(np.linspace(0,3*np.pi,100))+1]))[0])
   plt.figure()
   plt.plot(Neural('c:/py/doa/2017/output/weights',Patient(50,170,70,Gender.F))(np.array([np.sin(np.linspace(0,2*np.pi,5000))*1e-5+1e-5,np.sin(np.linspace(0,3*np.pi,5000))*1e-5+1e-5]))[0])
   plt.figure()
   plt.plot(Neural('c:/py/doa/2017/output/weights',Patient(50,170,70,Gender.F))(np.ones([2,5000]))[0])
