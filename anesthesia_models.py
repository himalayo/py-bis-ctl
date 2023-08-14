import numpy as np
import control as ct

class Patient:
    def __init__(self,age,height,weight):
        self.age = age
        self.height = height
        self.weight = weight

class Drug:
    def __init__(self,k,v):
        A = np.array([[-(k[0]+k[1]+k[2]),(v[1]/v[0])*k[3],(v[2]/v[0])*k[4]],
                      [(v[0]/v[1])*k[1],-k[3],0,0],
                      [(v[0]/v[2])*k[2],0,-k[3],0],
                      [k[5],0,0,k[5]]])
        B = np.array([[1/v[0]],[0],[0],[0]])
        C = np.array([0,0,0,1])
        self.ss = ct.ss(A,B,C,0)
        self.io = ct.ss2io(self.ss)

class Propofol(Drug):
    def __init__(self,p):
        def get_v(patient):
            v_1 = 5.1 - (0.0201 * (patient.age - 40)) + (0.072 * (patient.lbm - 55))
            v_2 = 9.82 - (0.0811 * (patient.age - 40)) + (0.108 * (patient.lbm - 55))
            v_3 = 5.42
            return v_1,v_2,v_3
            
        def get_k(patient):
            v = get_v(patient)

            k_10 = 2.6 - (0.0162*(patient.age-40)) +( 0.0191*(patient.lbm-55))
            k_12 = 2.05 - (0.301*(patient.age-40))
            k_13 = 0.076 - (0.00113*(patient.age-40))
            k_e0 = 0.595 - (0.007*(patient.age-40))

            k_10 /= v[0]
            k_12 /= v[0]
            k_13 /= v[0]
            k_21 = k_12*(v[0]/v[1])
            k_31 = k_13*(v[0]/v[2])

            return k_10, k_12, k_13, k_21, k_31, k_e0
        self.patient = p
        super().__init__(get_k(p),get_v(p))

class Remifentanil(Drug):
    def __init__(self,p):
        def get_v(patient):
            v_1 = 5.1 - (0.0201 * (patient.age - 40)) + (0.072 * (patient.lbm - 55))
            v_2 = 9.82 - (0.0811 * (patient.age - 40)) + (0.108 * (patient.lbm - 55))
            v_3 = 5.42
            return v_1,v_2,v_3
            
        def get_k(patient):
            v = get_v(patient)

            k_10 = 2.6 - (0.0162*(patient.age-40)) +( 0.0191*(patient.lbm-55))
            k_12 = 2.05 - (0.301*(patient.age-40))
            k_13 = 0.076 - (0.00113*(patient.age-40))
            k_e0 = 0.595 - (0.007*(patient.age-40))

            k_10 /= v[0]
            k_12 /= v[0]
            k_13 /= v[0]
            k_21 = k_12*(v[0]/v[1])
            k_31 = k_13*(v[0]/v[2])

            return k_10, k_12, k_13, k_21, k_31, k_e0
        self.patient = p
        super().__init__(get_k(p),get_v(p))