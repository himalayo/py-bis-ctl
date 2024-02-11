import numpy as np
from enum import Enum
mean_c =np.array([ 56.50928621,0.49630326,61.13655517,162.0409402])
std_c =np.array([15.38680592, 0.49998633, 9.23908289, 8.33116551])
class Gender(Enum):
    M=0
    F=1

    def parse(value: str|int|bool|float|None):
        match value:
            case str():
                val = Gender.M if value.lower().startswith('m') else Gender.F 
            case float():
                val = Gender.M if int(value) == 0 else Gender.F 
            case bool():
                val = Gender.F if value else Gender.M 
            case int():
                val = Gender.M if value == 0 else Gender.F 
            case None:
                val = Gender.M 
        return val

    def __str__(self):
        match self:
            case Gender.F:
                return 'F'
            case Gender.M:
                return 'M'
    def is_male(self):
        return self==Gender.M
    def is_female(self):
        return self==Gender.F
    def __bool__(self):
        return self.is_female() 
    def __int__(self):
        return int(self.__bool__())
    def __float__(self):
        return float(self.__int__())

def from_z(zs):
    n = (zs*std_c)+mean_c
    return Patient(n[0],n[3],n[2],Gender.parse(n[1]))

class Patient:
    def __init__(self,age: float, height: float, weight: float, gender: Gender):
        self.age = age
        self.height = height
        self.weight = weight 
        self.gender = gender
        if self.gender.is_female():
            self.lbm = (1.07*self.weight) - (148*((self.weight/self.height)**2)) 
        else: 
            self.lbm = (1.1*self.weight) - (((128*self.weight)**2)/(self.height**2))

        self.np = np.array([self.age,float(self.gender),self.weight,self.height])
        
        self.z = ((self.np-mean_c)/std_c).reshape(1,4)
    
    def __str__(self):
        return f"Age: {self.age}, Height: {self.height}, Weight: {self.weight}, Gender: {str(self.gender)}, np: {self.np}"


def test():
    rng = np.random.default_rng()
    genders = [Gender.parse('F'),Gender.parse('f'),Gender.parse(1),Gender.parse(1.0),Gender.parse(True)]
    assert(all(genders))
    assert(Gender(not all(genders)) == Gender.M)
    assert(not Gender.M)
    assert(Gender.F)
    assert(Gender(int(Gender.F)) == Gender.F)
    assert(Gender(int(Gender.M)) == Gender.M)
    patient = Patient(50,170,80,genders[0])
    print(patient)
    for _ in range(5):
        print(from_z(rng.normal(size=4)))

