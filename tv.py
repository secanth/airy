# produces data used for generating Figure 3

from params import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmap
import vandermonde
import scipy.integrate as integrate
from scipy.stats import pareto
from collections import deque
import time

# levels of separation to test
seps = np.pi * (np.e**np.linspace(2,-2,51))

# number of samples drawn for each data point on plot
TOTAL_SAMPS = 5000000
# samples are processed in batches of size batch
BATCH_SIZE = 100000

# generates set of points corresponding to one curve in the main figure
# the set contains, for every separation in seps, an estimate for the TV distance
# between a pair of superpositions of total_comps/2 Airy disks with that separation level
def main(total_comps):
	# main subroutine: outputs estimate for TV distance 
	# base_oracle: sample oracle for mixture of augmented pareto's approximating the two superpositions
	# weights_, centers_: parameters for the two superpositions of Airy disks
	def TV(base_oracle,weights_0,weights_1,centers_0,centers_1):
		# running counter for which batch we are processing
		counter = 0
		# running estimate for the TV distance
		result = 0
		# iterate over the batches
		for _ in range(TOTAL_SAMPS/BATCH_SIZE):
			# draw batch of samples from mixture of pareto's of size BATCH_SIZE
			points = base_oracle()
			# BATCH_SIZE x total_comps arrays consisting of distances from the pareto
			# mixture samples to all centers of the two superpositions of Airy disks
			dist_to_centers_0 = np.zeros((len(points),len(weights_0)))
			dist_to_centers_1 = np.zeros((len(points),len(weights_1)))
			# squared distance of the pareto mixture samples from the x-axis
			ysquared_diffs = points[:,1]**2
			# compute the entries of dist_to_centers_0
			for i in range(len(weights_0)):
				xdiff_to_center_i_0 = points[:,0] - centers_0[i]
				dist_to_centers_0[:,i] = np.sqrt(xdiff_to_center_i_0**2 + ysquared_diffs)
			# compute the entries of dist_to_centers_0
			for i in range(len(weights_1)):
				xdiff_to_center_i_1 = points[:,0] - centers_1[i]
				dist_to_centers_1[:,i] = np.sqrt(xdiff_to_center_i_1**2 + ysquared_diffs)

			# Boolean matrix indicating which points fall within the "augmented" part of the 
			# augmented Pareto around each of the centers
			unit_interval_points = np.hstack(((dist_to_centers_0 < 1.), (dist_to_centers_1 < 1.)))
			# pareto density (with parameter 2/3) evaluated at each sample with respect to each center
			# NOTE: 2*pi comes from polar coordinates
			pareto_densities_0 = pareto.pdf(dist_to_centers_0, 2/3.) / dist_to_centers_0 * weights_0 / (2 * np.pi)
			pareto_densities_1 = pareto.pdf(dist_to_centers_1, 2/3.) / dist_to_centers_1 * weights_1 / (2 * np.pi)
			pareto_densities = np.hstack((pareto_densities_0,pareto_densities_1))

			# density of "augmented" part of the augmented Pareto evaluated at each sample with respect to each center
			unit_interval_densities_0 = 1./dist_to_centers_0 * weights_0 / (2 * np.pi)
			unit_interval_densities_1 = 1./dist_to_centers_1 * weights_1 / (2 * np.pi)
			unit_interval_densities = np.hstack((unit_interval_densities_0,unit_interval_densities_1))

			# density of the mixture of augmented paretos at each of the points
			# NOTE: factor of /2 at the end is because proposal_densities is over both sets of centers
			proposal_densities = np.sum(pareto_densities * (1 - unit_interval_points)/2. + unit_interval_densities * unit_interval_points/2.,axis=1)/2
			
			# densities of the two superpositions of airy disks at each of the points
			D0_densities = np.sum(airy(dist_to_centers_0) * weights_0, axis=1)
			D1_densities = np.sum(airy(dist_to_centers_1) * weights_1, axis=1)

			# absolute difference in Radon-Nikodym derivatives between the two superpositions relative to the mixture of augmented paretos
			# NOTE: normalization constant pi comes from fact that integral of J_1(sqrt(x^2+y^2))/(x^2+y^2) over R^2 is pi
			ratio_diffs = np.abs((D1_densities - D0_densities)/proposal_densities) / np.pi

			# average absolute difference across this batch
			new_av = np.average(ratio_diffs)

			# update running average
			result = (result * counter + new_av)/(counter + 1.)
			counter += 1

		# output TV estimate (note TV = L1/2, hence factor of 2)
		return result/2.

	# define all 2*total_comps centers of the two superpositions of Airy disks 
	# to be interlacing, equidistant on x-axis
	def set_centers():
		centers = []
		if total_comps % 2 == 0:
			centers = range(-total_comps/2,total_comps/2)
		else:
			centers = range(-total_comps/2+1,total_comps/2+1)
		return centers

	# given set of all centers among two mixtures, returns centers and weights of two mixtures which moment match
	# by solving the appropriate Vandermonde system
	def set_weights(centers):
		# degree to which we moment-match
		d = total_comps - 1
		# solve (affine) vandermonde system to get the relative intensities for the two superpositions
		weights = vandermonde.solve_transpose(np.array(centers[:-1]), -np.array([centers[-1]**i for i in range(d)]))
		weights = list(weights) + [1]

		# parameters for the two superpositions
		centers_0 = []
		centers_1 = []
		weights_0 = []
		weights_1 = []

		# for each center i of either superposition, ids[i] is 0 if i belongs to first superposition, 1 otherwise
		ids = [0]*total_comps

		# the weights which are positive correspond to components of first superposition
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
		# normalize relative intensities
		for i in range(len(weights_0)):
			weights_0[i] /= total
		for i in range(len(weights_1)):
			weights_1[i] /= -total

		return centers_0,centers_1,weights_0, weights_1

	# (unnormalized) radial Airy point-spread function
	def airy(norm):
		return (bessel(1,norm)**2)/(norm**2)

	# generate BATCH_SIZE Haar-random unit vectors
	def random_unit_vectors():
	    cov = [[1,0],[0,1]]
	    x = np.random.multivariate_normal((0,0),cov,BATCH_SIZE)
	    norms = np.sqrt(x[:,0]**2 + x[:,1]**2)
	    return x/np.vstack((norms,norms)).T

	# outputs an oracle which, when called, outputs BATCH_SIZE iid draws from mixture of "augmented paretos"
	# whose weights and centers are the union of those of the two Airy superpositions
	# NOTE: an "augmented pareto" is a 2D radially-symmetric distribution whose radial distribution is 
	# given by  pareto (with parameter 2/3) concatenated with a uniform distribution over
	# the neighborhood of the origin over which the pareto distribution has zero mass
	def pareto_oracle(weights0,centers0,weights1,centers1):
		# given two sets of relative intensities (corresponding to the two Airy disk superpositions)
		# draw from either superposition with probability 1/2 times the corresponding relative intensity
		weights = weights0 + weights1
		weights = [weight/2. for weight in weights]
		# centers of the mixture of augmented paretos are the union of the centers of the two superpositions of Airy disks
		centers = centers0 + centers1
		def oracle():
			rs = pareto.rvs(2./3,size=BATCH_SIZE)
			# mask for which of the points to sample from [0,1] versus from pareto
			signs = np.random.random(BATCH_SIZE) < 1/2.
			# choose a random subset of the batch of samples to come from the "augmented" part of the pareto
			unit_interval_points = np.random.random(BATCH_SIZE)
			rs = rs * signs + unit_interval_points * (1 - signs)
			ids = np.random.choice(len(weights), BATCH_SIZE, p=weights)
			v = random_unit_vectors()
			xs = v[:,0] * rs + np.array(centers)[ids]
			ys = v[:,1] * rs
			out = np.vstack((xs,ys)).T
			return out
		return oracle

	# run TV on the appropriate pair of Airy superpositions, given level of separation sep
	def test(sep):
		centers = np.array(set_centers())*sep/2.
		centers_0,centers_1,weights_0, weights_1 = set_weights(centers)
		base_oracle = pareto_oracle(weights_0,centers_0,weights_1,centers_1)
		return TV(base_oracle,weights_0,weights_1,centers_0,centers_1)

	# for every level of separation sep, run test(sep) to produce a data point
	outs = [0]*len(seps)
	for i,sep in enumerate(seps):
		outs[i] = test(sep)
		# if TV goes below machine precision, no need to try calculating it
		if np.log(outs[i]) < -35.:
			break
    
    # output list of data points
	return outs

# different values of total_comps that we test
ks = [2, 4, 6, 12, 20, 30, 42, 56, 72, 90]
# final data matrix where the (i,j)-th entry is 
# TV between two superpositions of ks[i]/2 Airy disks at separation seps[j]
outs = np.zeros((len(ks),len(seps)))
for i in range(len(ks)):
	k = ks[i]
	# result is a list of length len(seps)
	result = main(k)
	for j in range(len(seps)):
		outs[i,j] = result[j]

# dump data matrix
with open('data/tvplot.json', 'w') as f:
	json.dump(outs.tolist(),f)
