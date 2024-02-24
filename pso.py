import tensorflow as tf

@tf.function
def build_swarm(population_size,dimension,minimum,maximum):
   return tf.random.uniform([population_size,dimension], minimum, maximum)
   
@tf.function
def start_velocities(population_size,dimension,minimum,maximum):
    return tf.random.uniform([population_size,dimension],minimum,maximum)

@tf.function
def get_randoms(dimension):
    return tf.random.uniform([2, dimension], 0,1)

@tf.function
def update_p_best(fn,x,f_p,p,**kwargs):
    f_x = fn(x,**kwargs)
    mask = f_x<f_p
    new_p = tf.where(tf.repeat(mask[:,tf.newaxis],x.shape[1],1),x,p)
    new_f_p = tf.where(mask,f_x,f_p)
    return new_p,new_f_p

@tf.function
def update_g_best(f_p,p):
    return p[tf.math.argmin(input=f_p)]

@tf.function
def random(pos,lower_bound,upper_bound):
    lb = tf.ones(pos.shape)*lower_bound
    ub = tf.ones(pos.shape)*upper_bound
    return tf.where(tf.math.logical_or(pos<lb,pos>ub),tf.random.uniform(pos.shape,lower_bound,upper_bound),pos) 

@tf.function
def step(fn,f_p,v,b,c1,c2,p,g,x,minimum,maximum,population_size,dimension,**kwargs):
    rnds = get_randoms(dimension)
    r1 = rnds[0]
    r2 = rnds[1]
    new_v = (
        b * v
        + c1 * r1 * (p - x)
        + c2 * r2 * (g - x)
    )
    new_x = random(x+new_v,minimum,maximum)
    new_p,new_f_p = update_p_best(fn,new_x,f_p,p,**kwargs)
    new_g = update_g_best(new_f_p,new_p)
    tf.print(new_g,tf.math.reduce_min(new_f_p))
    return new_v,new_x,new_p,new_f_p,new_g

@tf.function
def optimize(fitness_fn,pop_size=100,dim=2,b=0.9,c1=0.8,c2=0.5,x_min=-1,x_max=1,max_iter=300,tol=1000,**kwargs):
        x = build_swarm(pop_size,dim,x_min,x_max)
        p = x
        f_p = fitness_fn(x,**kwargs)
        g = p[tf.math.argmin(input=f_p)]
        v = start_velocities(pop_size,dim,x_min,x_max)
        i = 0
        while tf.math.reduce_min(f_p) > tol and i<max_iter:
            v,x,p,f_p,g = step(fitness_fn,f_p,v,b,c1,c2,p,g,x,x_min,x_max,pop_size,dim,**kwargs)
            i += 1
            if i%20 == 0:
                v = start_velocities(pop_size,dim,x_min,x_max)
            if i%50 == 0:
                tf.print('Restarting...')
                x = build_swarm(pop_size,dim,x_min,x_max)
                p = x
                f_p = fitness_fn(x,**kwargs)
                g = p[tf.math.argmin(input=f_p)]
                v = start_velocities(pop_size,dim,x_min,x_max)

        return g


