import multiprocessing as processing
from multiprocessing import pool,Process
import math
import numpy as np
import gzip
import pickle
import sys
from scipy import integrate
import time,random
import h5py
import threading
import os.path
import _pickle as cpickle
from scipy import spatial
from multiprocessing import Process

def log10(x):
    if x > 0:
        return math.log10(x)
    else:
        return -np.inf


def exp(x):
    try:
        return math.exp(x)
    except:
        return np.inf


def Mpeak_log_to_Vpeak_log(Mpeak_log):
    return 0.3349 * Mpeak_log - 1.672


G = 4.301 * 10 ** (-9)
cons = (4 * G * np.pi / (3 * (1 / 24 / (1.5 * 10 ** (11))) ** (1 / 3))) ** 0.5


def calculate_v_dispersion(Mh):
    return Mh ** (1 / 3) * cons


exp = np.vectorize(exp)
log10 = np.vectorize(log10)

plot_path = "/Users/caojunzhi/Downloads/upload_201903_Jeremy/"

# data path

if os.path.isdir("/Volumes/SSHD_2TB") == True:
    print("The code is on Spear of Adun")

    ## Move to Data_10TB
    data_path = "/Volumes/Data_10TB/"

elif os.path.isdir("/mount/sirocco1/jc6933/test") == True:
    data_path = "/mount/sirocco2/jc6933/Data_sirocco/"
    print("The code is on Sirocco")

# Kratos
elif os.path.isdir("/home/jc6933/test_kratos") == True:
    data_path = "/mount/kratos/jc6933/Data/"
    print("The code is on Kratos")

# Void Seeker
elif os.path.isdir("/home/jc6933/test_Void_Seeker") == True:
    data_path = "/mount/Void_Seeker/Data_remote/"
    print("The code is on Void Seeker")

else:
    print("The code is on local")
    data_path = "/Volumes/Extreme_SSD/Data/"

print("data_path %s" % data_path)

save_path = data_path + "KD/"

class KD_search():
    def __init__(self):
        #read data:
        hf_Bolshoi = h5py.File(data_path + "C250_with_concentration.h5", "r")

        data_Bolshoi = np.array(hf_Bolshoi["dataset"])[:, 1:7]
        Mpeak = np.array(hf_Bolshoi["Mpeak"])
        Mvir = np.array(hf_Bolshoi["M_vir"])
        Vpeak = np.array(hf_Bolshoi["Vpeak"])
        upid = np.array(hf_Bolshoi["upid"], dtype=int)
        halo_concentration = np.array(hf_Bolshoi["halo_concentration"], dtype=float)
        Acc_Rate_Inst = np.array(hf_Bolshoi["Acc_Rate_Inst"], dtype=float)

        hf_Bolshoi.close()

        Mvir_log = log10(Mvir)

        pkl_file = open(data_path + "R_vir_z0_C250.pkl", 'rb')
        R_vir = pickle.load(pkl_file)
        pkl_file.close()

        pkl_file = open(data_path + "Mpeak/" + "Ms_log_C250_M_peak_combined_v1.pkl", 'rb')
        Ms_all_log = pickle.load(pkl_file)
        pkl_file.close()

        select = random.sample(range(0, len(R_vir) - 1), 100000)

        Mh_rate = Acc_Rate_Inst
        fb = 0.16
        Ms_rate = fb * Mh_rate

        sSFR = Ms_rate / Ms_all_log

        data_250 = data_Bolshoi[:, 0:3]

        """

        print("Constructing KD tree")

        tree_250 = spatial.KDTree(data_250)

        raw_250 = pickle.dumps(tree_250)

        # save it:

        file_path_250 = save_path + "kd_C250.pkl"

        max_bytes = 2 ** 31 - 1

        print("saving Bolshoi")

        ## write
        bytes_out = pickle.dumps(raw_250)
        with open(file_path_250, 'wb') as f_out:
            for idx in range(0, len(bytes_out), max_bytes):
                f_out.write(bytes_out[idx:idx + max_bytes])

        ##


        

        """
        print("reading tree")

        file_path_250 = save_path + "kd_C250.pkl"

        pkl_file = open(file_path_250, 'rb')
        raw = cpickle.load(pkl_file)
        pkl_file.close()

        tree_load_250 = cpickle.loads(raw)
        self.tree_load_250 = tree_load_250



        # calculate density: Be careful about the periodic condition:
        mask_cen = upid < 0

        ## Only calculate trees with halo mass bigger than 11.0 and in boundary:
        x = data_250[:, 0]
        y = data_250[:, 1]
        z = data_250[:, 2]

        self.x = x
        self.y = y
        self.z = z

        # For everything!
        mask = upid<0
        self.data_250_central = data_250[mask,:]

        N_tot = len(self.data_250_central[:, 0])
        self.N_tot = N_tot
        self.N = N_tot
        self.data_250 = data_250

        self.Ms_all_log = Ms_all_log

        self.dict_results = {}
        self.dict_results_mass = {}

    def calculate_density(self,N_thread,N_start,N_stop):
        density_array = []
        density_array_mass = []
        for i in range(N_start, N_stop):
            if i % 100 == 0:
                print("Doing %d of %d for thread %d" % (i - N_start, N_stop - N_start, N_thread))

            index = self.tree_load_250.query_ball_point(self.data_250_central[i, :], r=10)

            index = [item for item in index if item not in [i]]
            Ms_tot = np.nansum(10 ** self.Ms_all_log[index])
            density_array_mass.append(Ms_tot)
            density_array.append(len(index))

        density_array = np.array(density_array)
        density_array_mass = np.array(density_array_mass)

        self.dict_results[str(N_thread)] = density_array
        self.dict_results_mass[str(N_thread)] = density_array_mass


model = KD_search()


input = sys.argv
try:
    thread_to_use = int(input[1])

except:
    thread_to_use = 12


print("Total threads %d and total numbers %d"%(thread_to_use,model.N_tot))

bin_size = model.N_tot//thread_to_use

my_pool = []

for ni in range(0,thread_to_use):
    if ni<thread_to_use-1:
        pi = Process(target=model.calculate_density, args=(ni, ni * bin_size, ni * bin_size + bin_size))
        my_pool.append(pi)
        pi.start()

    else:
        pi = Process(target=model.calculate_density, args=(ni, ni * bin_size, model.N))
        my_pool.append(pi)
        pi.start()

# join and wait until all threads are done:
for ni in range(0,thread_to_use):
    my_pool[ni].join()

##Done
print("All threads done")
# add each chunks together

density_array_all = []
for x in range(0, thread_to_use):
    density_array_all.extend(model.dict_results[str(x)])
density_array_all = np.array(density_array_all)


# save in txt is good since the size is not big:
np.savetxt(data_path+"KD/"+"density_C250_multithread_v1.txt",density_array_all)



density_array_mass_all = []
for x in range(0, thread_to_use):
    density_array_mass_all.extend(model.dict_results_mass[str(x)])
density_array_mass_all = np.array(density_array_mass_all)


# save in txt is good since the size is not big:
np.savetxt(data_path+"KD/"+"density_C250_multithread_mass_v1.txt",density_array_mass_all)




