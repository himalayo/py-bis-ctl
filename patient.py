import numpy as np
from enum import Enum

class Gender(Enum):
    M=0,
    F=1

    def __init__(self,value: str|int|bool|float|None):
        match value:
            case str():
                self = Gender.M if value.lower().startswith('m') else Gender.F
            case float():
                self = Gender.M if int(value) == 0 else Gender.F
            case bool():
                self = Gender.F if value else Gender.M
            case int():
                self = Gender.M if value == 0 else Gender.F
            case None:
                self = Gender.M

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

class Patient:
    def __init__(self,age: float, height: float, weight: float, gender: Gender):
        self.age = age
        self.height = height
        self.weight = weight 
        self.gender = gender
        self.np = np.array([self.age,float(self.gender),self.height,self.weight])