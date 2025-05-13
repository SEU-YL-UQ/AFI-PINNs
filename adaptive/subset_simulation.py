from random import sample
import numpy as np 
import matplotlib.pyplot as plt 
# from prior_density import Uniform
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

class Subset_simulation:
    """This class implements subset simulation manaully"""
    def __init__(self, prior, final_level, p0 = 0.1) -> None:
        self.p0 = p0 
        self.prior = prior 
        self.final_level = final_level 
    
        

    def MMA_uniform(self, seeds, power_function, current_level, num_samples):
        dim = seeds.shape[1]
        num = seeds.shape[0]
        add_samples = np.zeros((0, dim))
        num_add_samples = np.zeros(num)
        currentx = seeds
        failed_index = num_add_samples<num_samples
        while np.sum(failed_index)>0:
            # print(len(add_samples))
            currentx = currentx[failed_index]
            proposedx = self.prior.sample(np.sum(failed_index))
            acc = np.exp(-0.5*(proposedx**2 - currentx**2))
            acc[acc>1] = 1
            u = np.random.rand(np.sum(failed_index), dim)
            index = (u > acc)
            proposedx[index] = currentx[index]
            select_index = (power_function(proposedx)>current_level)
            currentx[select_index] = proposedx[select_index]
            num_add_samples[select_index] = num_add_samples[select_index] + 1
            failed_index = num_add_samples<num_samples
            num_add_samples = num_add_samples[failed_index]
            add_samples = np.append(add_samples, proposedx[select_index], axis = 0)
        return add_samples
    
    def MMA_normal(self, seeds, power_function, current_level, num_samples):
        dim = len(seeds.squeeze())
        add_samples = seeds.reshape(1,-1) 
        i = 1
        currentx = add_samples[i-1]
        while i <= num_samples:
            proposedx = np.random.multivariate_normal(currentx, np.eye(dim), 1)
            acc = np.exp(-0.5*(proposedx**2 - currentx**2))
            acc[acc>1] = 1
            u = np.random.rand(dim)
            index = (u > acc).squeeze()
            proposedx[0, index] = currentx[index]
            if power_function(proposedx) > current_level:
                add_samples = np.vstack([add_samples, proposedx])
                currentx = proposedx.squeeze()
                i += 1
            else:
                currentx = currentx
        return add_samples[1:]

    def MMA_uniform1(self, seeds, power_function, current_level, num_samples):
        dim = seeds.shape[1]
        num = seeds.shape[0]
        add_samples = np.zeros((0, dim))
        currentx = seeds.copy()
        while add_samples.shape[0]< num*num_samples:
            print("\r", end="")
            print('current number of samples: {}/{}'.format(add_samples.shape[0], num*num_samples), end = ' ')
            sys.stdout.flush()
            proposedx = self.prior.sample(num)
            acc = np.exp(-0.5*(proposedx**2 - currentx**2))
            acc[acc>1] = 1
            u = np.random.rand(num, dim)
            index = (u > acc)
            proposedx[index] = currentx[index]
            select_index = (power_function(proposedx)>current_level)
            currentx[select_index] = proposedx[select_index]
            select_index[np.sum(index, axis = 1) == dim] = False
            add_samples = np.append(add_samples, proposedx[select_index], axis = 0)
        return add_samples[:np.int32(num*num_samples)]
    

    def sample(self, num_samples, power_function, tol_p):
        if self.prior.name == 'uniform':
            MH = self.MMA_uniform1
        else:
            MH = self.MMA_normal
        samples = self.prior.sample(num_samples)
        power_values = power_function(samples)
        num_seeds = np.floor(samples.shape[0] * self.p0).astype(np.int32)
        num_failure_samples = np.sum(power_values > self.final_level)
        print('Initial samples:', num_samples)
        level_set = 0
        failure_p = 1
        while num_seeds > num_failure_samples:
            self.iter = 0
            seeds = samples[np.argsort(power_values)[::-1]][:num_seeds]
            power_values = np.sort(power_values)[::-1]
            current_level = np.mean(power_values[num_seeds - 1:num_seeds + 1])
            # add_samples = np.apply_along_axis(MH, arr = seeds, axis = 1, power_function = power_function, current_level = current_level, num_samples = 1/self.p0 - 1)
            add_samples = MH(seeds, power_function, current_level, 1/self.p0 - 1)
            # add_samples = add_samples.reshape(-1, samples.shape[1])
            samples = np.vstack([seeds, add_samples])
            add_power_values = power_function(add_samples)
            power_values = np.append(power_values[:num_seeds], add_power_values)
            num_failure_samples = np.sum(power_values > self.final_level)
            level_set += 1
            print("[Initial number of failure samples: {}][Total number of failure samples: {}]".format(num_failure_samples, num_seeds))
            if level_set > 5:
                break
        failure_p = self.p0**level_set * num_failure_samples/samples.shape[0]
        return samples[power_values>self.final_level], failure_p








    


    
    
    

    

    