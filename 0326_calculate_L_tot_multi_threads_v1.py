import math
import numpy as np
import gzip
import pickle, _pickle
from scipy import integrate
import time
import h5py
import threading
import os.path


# show plots:
# get_ipython().run_line_magic('matplotlib', 'inline')


def jack_knife_error_bar(n_jack, samples):
    # return 1*n_jack array:
    error_bar = []
    bin = len(samples) // n_jack
    for i in range(0, n_jack):
        samples_i = list(samples)
        del samples_i[i * bin:i * bin + bin]
        error_bar.append((np.percentile(samples_i, 84) - np.percentile(samples_i, 16)) / 2)
    error_bar = np.array(error_bar)
    return error_bar / 10


def bootstrap_scatter_err(samples):
    if len(samples) < 3:
        return np.nan
    else:
        mask_finite = np.isfinite(samples)
        samples = samples[mask_finite]
        index_all = range(len(samples))
        err_all = []
        N = 100
        for i in range(0, N):
            index_choose = np.random.randint(0, len(samples) - 1, len(samples))
            # k_i = np.nanstd(samples[index_choose])
            k_i = np.percentile(samples[index_choose], 84) - np.percentile(samples[index_choose], 16)
            k_i = k_i / 2
            err_all.append(k_i)
        err_all = np.array(err_all)
        if len(samples) < 0:
            err_all = np.nan

        return err_all


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


def _generic_open(f, buffering=100000000):
    """
    Returns
    -------
    fp : file handle
    need_to_close : bool
    """
    if hasattr(f, 'read'):
        return f, False
    else:
        if f.endswith('.gz'):
            fp = gzip.open(f, 'r')
        else:
            fp = open(f, 'r', int(buffering))
        return fp, True

"""

# Convert between L and magnitude:

def Sersic_fit(phi_alpha, M):
    rou = 0.344

    alpha = 1.665
    beta = 0.255
    gamma = 0.296

    M_gamma = 2.7031

    M_s = 0.0094

    phi_gamma = 0.675

    # phi_alpha =(rou-phi_gamma*M_gamma*special.gamma(1+gamma))/(M_s*special.gamma((1+alpha)/beta)/special.gamma(alpha/beta))

    return phi_alpha * beta * (M / M_s) ** alpha * (
            exp(-(M / M_s) ** beta) / special.gamma(alpha / beta)) + phi_gamma * (M / M_gamma) ** gamma * (
               exp(-(M / M_gamma)))


"""

## Double Schechter function:

# calculate L_lim:
M_sun = 4.53
M_apparent_lim = 24

# distance in pc:
distance = 324.67 * 1e6

# apparent magnitude
M_lim = 24 - 5 * (log10(distance) - 1)
# print(M_lim)
M_sun = 4.53
# L_lim in unit of L_sun

L_lim = 10 ** ((M_lim - M_sun) / (-2.5))


# r band magnitude
def M_to_L_r_band(Mr):
    return 10 ** ((Mr - M_sun) / (-2.5))


# Double Schechter function:

def Schechter_function(M, phi_s1, phi_s2, M_s, alpha_1, alpha_2):
    return 0.4 * 2.3 * np.exp(-10 ** (-0.4 * (M - M_s))) * (
            phi_s1 * 10 ** (-0.4 * (M - M_s) * (alpha_1 + 1)) + phi_s2 * 10 ** (-0.4 * (M - M_s) * (alpha_2 + 1)))


def box_smooth(data_array):
    data_array = np.array(data_array)
    N = len(data_array)

    data_smooth = []

    for i in range(0, N):
        data_i = data_array[int(np.maximum(i - 1, 0)):int(np.minimum(i + 2, N))]
        # print(np.nanmean(data_i))

        data_smooth.append(np.nanmean(data_i))

    data_smooth = np.array(data_smooth).ravel()
    data_smooth[0] = np.nanmedian(data_array[:1])
    return data_smooth


class sSFR():

    def __init__(self, kwargs):
        self.kwargs = kwargs
        phi_s1, phi_s2, M_s, alpha_1, alpha_2 = kwargs["phi_s1"], kwargs["phi_s1"], kwargs["M_s"], kwargs["alpha_1"], \
                                                kwargs["alpha_2"]

        # Let's match Mr rather than L:
        Mr_target = np.linspace(-24, -12, 100)[::-1]

        y_schechter = Schechter_function(M=Mr_target, phi_s1=phi_s1, phi_s2=phi_s2, M_s=M_s, alpha_1=alpha_1,
                                         alpha_2=alpha_2)

        y_schechter_log = log10(y_schechter)
        # plt.plot(Mr_target,y_schechter_log,"ko")
        # plt.gca().invert_xaxis()

        # fit using a poly
        poly = np.poly1d(np.polyfit(Mr_target, y_schechter_log, 10))

        self.poly = poly

        # dictionary to save results from each thread:
        self.dict_results = {}

    def cumc_L_mr(self, Mr):

        none = 1
        # plt.plot(Mr_target,poly(Mr_target),"r")
        # plt.show()

        # Let's do the cumulative one:

        return integrate.quad(lambda x: 10 ** self.poly(x), -24, Mr)[0]

    def read_C250(self):
        # read sSFR:

        hf_Bolshoi = h5py.File(data_path + "C250_with_concentration.h5", "r")

        data_Bolshoi = np.array(hf_Bolshoi["dataset"])[:, 1:4]
        Mpeak = np.array(hf_Bolshoi["Mpeak"])
        Mvir = np.array(hf_Bolshoi["M_vir"])
        Vpeak = np.array(hf_Bolshoi["Vpeak"])
        self.upid = np.array(hf_Bolshoi["upid"], dtype=int)
        self.pid = np.array(hf_Bolshoi["pid"], dtype=int)
        self.id_C250 = np.array(hf_Bolshoi["id"], dtype=int)
        halo_concentration = np.array(hf_Bolshoi["halo_concentration"], dtype=float)
        Acc_Rate_Inst = np.array(hf_Bolshoi["Acc_Rate_Inst"], dtype=float)

        hf_Bolshoi.close()

        Mh_log = log10(Mvir)

        pkl_file = open(data_path + "R_vir_z0_C250.pkl", 'rb')
        R_vir = pickle.load(pkl_file)
        pkl_file.close()

        # read Ms from Vpeak abundance matching

        pkl_file = open(data_path + "Vpeak/" + "Ms_array_C250_Vpeak_v1.pkl", 'rb')
        self.Ms_all_log = pickle.load(pkl_file)
        pkl_file.close()

        L_path = data_path + "LHAM/" + "L_array_log_C250.pkl"

        pkl_file = open(L_path, 'rb')
        L_log_array = pickle.load(pkl_file)
        pkl_file.close()



        mask_central = (self.upid < 0) & (self.Ms_all_log > 9.5)
        self.id_C250_central = self.id_C250[mask_central]
        self.id_C250_central = self.id_C250_central
        self.N = len(self.id_C250_central)
        self.L_array = 10 ** L_log_array

    def calculate_density(self,N_thread,N_start,N_stop):

        L_tot_log_array = []

        for i in range(N_start,N_stop):
            index_i = np.where(self.pid == self.id_C250_central[i])
            if len(index_i) > 0:
                L_tot_log_array.append(log10(np.nansum(self.L_array[index_i])))
            else:
                L_tot_log_array.append(0)

            if i % 100 == 0:
                print("Doing %d of %d for thread %d" % (i, N_stop-N_start,N_thread))
        L_tot_log_array = np.array(L_tot_log_array)
        self.dict_results[str(N_thread)] = L_tot_log_array


#### Parameters:


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

# check schechter:

kwargs = {"phi_s1": None, "phi_s2": None, "M_s": None, "alpha_1": None, "alpha_2": None}

# set up initial values:

kwargs["phi_s1"] = 1.56 * 0.01
kwargs["phi_s2"] = 0.62 * 0.01
kwargs["M_s"] = -20.04
kwargs["alpha_1"] = -0.17
kwargs["alpha_2"] = -1.52

## Here we use logMh not logMs, and we do not need AM anymore!

model = sSFR(kwargs=kwargs)

save_path = data_path + "KD/"
### C250:
model.read_C250()

# calculate:
# model.calculate_density(N_thread=0,N_start=0,N_stop=300)

thread_to_use = 12

bin_size = model.N//thread_to_use
my_pool = []

for ni in range(0,thread_to_use):

    if ni<thread_to_use-1:
        my_pool.append(threading.Thread(target=model.calculate_density,args=(ni,ni*bin_size,ni*bin_size+bin_size)))

    else:
        my_pool.append(threading.Thread(target=model.calculate_density,args=(ni,ni*bin_size,model.N)))

# start
for x in range(0, thread_to_use):
    print("Start thread %d" % x)
    my_pool[x].start()



# wait until all threads are done
for x in range(0, thread_to_use):
    my_pool[x].join()


# add each chunks together

L_tot_log_array_final = []
for x in range(0, thread_to_use):
    L_tot_log_array_final.extend(model.dict_results[str(x)])
L_tot_log_array_final = np.array(L_tot_log_array_final)


# save in txt is good since the size is not big:
np.savetxt(data_path+"LHAM/"+"L_tot_log_array_C250_bigger_than_95_multithread.txt",L_tot_log_array_final)





