import numpy as np
import patient
import controllers
import json
import os
import gc

def infinite_range(x=0):
    while True:
        x += 1
        yield x 

if __name__ == '__main__':
    rng = np.random.default_rng()
    previous_cases = filter(lambda x: x.endswith('.json'),os.listdir('collected/'))
    last_case = 0 if not previous_cases else sorted([int(case[len('case_'):-len('.json')]) for case in previous_cases])[-1]
    cases = infinite_range(last_case)
    for case in cases:
        try:
            print(case)
            p = patient.from_z(rng.normal(size=4))
            print(p)
            mdl = controllers.NNET(p)
            pid = controllers.get_PID(mdl,0.5).x
            ctl = controllers.PID(p,mdl,*pid)
            
            data = [] 
            x = 0.98
            for i in range(150):
                x = ctl.update(0.5,x)
                data.append({'err':float(np.copy(ctl.err[0])),'prop':float(np.copy(ctl.prop[0,-1,0])),'remi':float(np.copy(ctl.remi[0,-1,0]))})
                print(p.np,data[-1])
            c = {'id':case,'patient':list(p.np),'data':data}
            with open(f'collected/case_{case}.json','w') as case_file:
                json.dump(c,case_file)
            del pid
            del ctl
            del mdl
            del c
            del data
            gc.collect()
        except:
            break

