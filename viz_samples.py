# creates samples used for generating Figure 2

from params import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmap
import vandermonde
import scipy.integrate as integrate
from scipy.stats import pareto
from collections import deque

# parameter corresponding to diffraction limit
diff_limit = np.pi/2

# level of separation relative to diffraction limit
factor = 0.2

# generates a matrix whose num rows are random unit vectors
def random_unit_vector(num=1):
    cov = [[1,0],[0,1]]
    x = np.random.multivariate_normal((0,0),cov,num)
    norms = np.sqrt(x[:,0]**2 + x[:,1]**2)
    return x/np.vstack((norms,norms)).T

# samples N radii according to radial Airy psf by rejection sampling via cauchy distribution
def sample_airy_rs(N):
    # we will build up the list of N samples in batches of size ~N/10
    batch = N/10
    out = np.zeros((N,))
    counter = 0
    while counter < N:
        # hand-tuned scaling factor so that N1 Cauchy samples will yield roughly a batch of N/10 airy samples
        M = 5.3
        N1 = int(M*batch)
        cauchy = np.abs(np.random.standard_cauchy(N1))
        # rejection sample to get samples from airy
        result = cauchy[np.random.random(N1) < (bessel(1,cauchy)**2)*(np.pi * (1/cauchy + cauchy))/M]
        n1 = len(result)
        # add samples to output matrix
        out[counter:min(counter+n1,N)] = result[:min(n1,N-counter)]
        # update total number of airy samples produced
        counter += n1
    out = out[:N]
    return out

# given radii rs and an x-coordinate center, produce N 2D airy samples centered at (center,0) with those radii
def sample_airy(rs,center):
    N = len(rs)
    return random_unit_vector(N)*rs[:,np.newaxis] + np.array([center,0])

# same as in tv.py
def set_centers(total_comps):
    centers = []
    if total_comps % 2 == 0:
        centers = range(-total_comps/2,total_comps/2)
    else:
        centers = range(-total_comps/2+1,total_comps/2+1)
    return np.array(centers)

# same as in tv.py
def set_weights(centers):
    total_comps = len(centers)
    d = total_comps - 1
    weights = vandermonde.solve_transpose(np.array(centers[:-1]), -np.array([centers[-1]**i for i in range(d)]))
    centers_0 = []
    centers_1 = []
    weights_0 = []
    weights_1 = []
    weights = list(weights) + [1]
    ids = [0]*total_comps

    for i in range(total_comps):
        if weights[i] > 0:
            weights_0.append(weights[i])
            centers_0.append(centers[i])
            ids[i] = 0
        else:
            weights_1.append(weights[i])
            centers_1.append(centers[i])
            ids[i] = 1
    total = sum(weights_0)
    for i in range(len(weights_0)):
        weights_0[i] /= total
    for i in range(len(weights_1)):
        weights_1[i] /= -total

    return centers_0,centers_1,weights_0, weights_1

# draw total_N samples from the two pi*factor-separated superpositions, which together have a total of twok components
# and plot the corresponding 1D histogram of x-coordinates for both superpositions
def test(total_N,twok):
    centers_0 ,centers_1, weights_0, weights_1 = set_weights(set_centers(twok) * diff_limit*factor)
    # sample from the first mixture
    rs_0 = sample_airy_rs(total_N)
    samples_0 = sample_airy(rs_0,0.)
    sampled_centers_0 = np.array(centers_0)[np.random.choice(len(weights_0), total_N, p=weights_0)]
    samples_0[:,0] += sampled_centers_0

    # sample from the second mixture
    rs_1 = sample_airy_rs(total_N)
    samples_1 = sample_airy(rs_1,0.)
    sampled_centers_1 = np.array(centers_1)[np.random.choice(len(weights_1), total_N, p=weights_1)]
    samples_1[:,0] += sampled_centers_1

    # dump samples
    with open('data/viz_samples_0_%d.json' % total_N, 'r') as f:
        samples_0 = json.load(f)
    with open('data/viz_samples_1_%d.json' % total_N, 'r') as f:
        samples_1 = json.load(f)

test(2000,3)
test(200000,3)
test(20000000,3)



