import matplotlib.pyplot as plt
import numpy as np
import control as ct
from patient import Patient, Gender

class Drug:
    def __init__(self,k,v):
        A = np.array([[-(k[0]+k[1]+k[2]),(v[1]/v[0])*k[3],(v[2]/v[0])*k[4],0],
                      [(v[0]/v[1])*k[1],-k[3],0,0],
                      [(v[0]/v[2])*k[2],0,-k[4],0],
                      [k[5],0,0,-k[5]]])
        B = np.array([[1/v[0]],[0],[0],[0]])
        C = np.array([0,0,0,1])
        self.ss = ct.ss(A,B,C,0)
        self.io = ct.ss2io(self.ss)

class Propofol(Drug):
    def __init__(self,p: Patient):
        def get_v(patient):
            v_1 = 4.27
            v_2 = 18.9 - 0.391*(patient.age-53) 
            v_3 = 238
            return v_1,v_2,v_3
            
        def get_k(patient):
            v = get_v(patient)

            k_10 = 0.443 + 0.0107*(patient.weight-77) -0.0159*(patient.lbm)+0.0062*(patient.height-177)
            k_12 = 0.302 - 0.0056*(patient.age-53) 
            k_13 = 0.196
            k_21 = (1.29-0.024*(patient.age-53))/v[1]
            k_31 = 0.0035
            k_e0 = 0.456


            return k_10, k_12, k_13, k_21, k_31, k_e0
        self.patient = p
        super().__init__(get_k(p),get_v(p))

class Remifentanil(Drug):
    def __init__(self,p):
        def get_v(patient):
            v_1 = 5.1 - (0.0201 * (patient.age - 40)) + (0.072 * (patient.lbm - 55))
            v_2 = 9.82 - (0.0811 * (patient.age - 40)) + (0.108 * (patient.lbm - 55))
            v_3 = 5.42
            return -v_1,v_2,v_3
            
        def get_k(patient):
            v = get_v(patient)

            k_10 =(2.6 - 0.0162*(patient.age-40) + 0.0191*(patient.lbm-55))/-v[0]
            k_12 = -(2.05 - (0.301*(patient.age-40)))/-v[0]
            k_13 = -(0.076 - (0.00113*(patient.age-40)))/-v[0]
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

    