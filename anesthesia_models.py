import matplotlib.pyplot as plt
import numpy as np
from patient import Patient, Gender

class Drug:
    def __init__(self,ks,vs):
        self.k = ks
        self.v = vs
        self.x = np.zeros(4)
        self.A = np.array([[-(ks[0]+ks[1]+ks[2]),(vs[1]/vs[0])*ks[1],ks[2]*(vs[2]/vs[0]),0],
                           [(vs[0]/vs[1])*ks[3],-ks[1],0,0],
                           [(vs[0]/vs[2])*ks[4],0,-ks[2],0],
                           [ks[5],0,0,-ks[5]]])
        self.B = np.array([[1/vs[0]],[0],[0],[0]])
        r = np.concatenate([self.B,np.dot(self.A,self.B),np.dot(self.A**2,self.B),np.dot(self.A**3,self.B)],axis=1)
        if np.linalg.matrix_rank(r) != 4:
            print("omegalul")
    def model_step(self,u):
        c0_k0 = self.x[0]*self.k[0]
        c0_k1 = self.x[0]*self.k[1]
        c0_k2 = self.x[0]*self.k[2]
        c0_k5 = self.x[0]*self.k[5]
        c1_k3 = self.x[1]*self.k[3]
        c2_k4 = self.x[2]*self.k[4]
        c3_k5 = self.x[3]*self.k[5]
        return (u + (c1_k3*(self.v[2]/self.v[0])) - c0_k1 + (c2_k4*(self.v[1]/self.v[0])) -c0_k2 - c0_k0, 
		        (c0_k1*(self.v[0]/self.v[1])) - c1_k3,
		        (c0_k2*(self.v[0]/self.v[2])) - c2_k4,
		        c0_k5 - c3_k5)

    def __call__(self,d):
        self.x += self.model_step(d)
        return self.x[3]
class Propofol(Drug):
    def __init__(self,p: Patient):
        def get_v(self,patient):
            return 0.228,0.463,2.893 
        
        def get_k(self,patient):
            v = get_v(self,patient)
            k_10 = 0.119
            k_12 = 0.112
            k_13 = 0.042 
            k_21 = 0.055 
            k_31 = 0.0033
            k_e0 = 0.26

            return k_10, k_12, k_13, k_21, k_31, k_e0
     
        self.patient = p
        super().__init__(get_k(self,p),get_v(self,p))

class Remifentanil(Drug):
    def __init__(self,p):
        def get_v(patient):
            v_1 = 5.1 - (0.0201 * (patient.age - 40)) + (0.072 * (patient.lbm - 55))
            v_2 = 9.82 - (0.0811 * (patient.age - 40)) + (0.108 * (patient.lbm - 55))
            v_3 = 5.42
            return v_1,v_2,v_3
            
        def get_k(patient):
            v = get_v(patient)

            k_10 =(2.6 - (0.0162*(patient.age-40)) + (0.0191*(patient.lbm-55))/v[0])
            k_12 = (2.05 - (0.0301*(patient.age-40)))/v[0] 
            k_13 = (0.076 - (0.00113*(patient.age-40)))/v[0]
            k_e0 = 0.595 - (0.007*(patient.age-40))

            k_21 = k_12*(v[0]/v[1])
            k_31 = k_13*(v[0]/v[2])

            return k_10, k_12, k_13, k_21, k_31, k_e0
        self.patient = p
        super().__init__(get_k(p),get_v(p))

def test():
    print(Remifentanil(Patient(50,170,80,Gender.F)).ss.A)
    print(Remifentanil(Patient(50,170,80,Gender.F)).ss.B)
    print(Propofol(Patient(50,170,80,Gender.F)).ss.A)
    print(Propofol(Patient(50,170,80,Gender.F)).ss.B)
    plt.figure()
    plt.plot(ct.step_response(Remifentanil(Patient(50,170,80,Gender.F)).io).outputs)
    plt.plot(ct.step_response(Propofol(Patient(50,170,80,Gender.F)).io).outputs)
    plt.figure()
    plt.plot(ct.impulse_response(Remifentanil(Patient(50,170,80,Gender.F)).io).outputs)
    plt.plot(ct.impulse_response(Propofol(Patient(50,170,80,Gender.F)).io).outputs)

    
