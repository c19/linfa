use crate::KMeansParamsError;

use super::init::KMeansInit;
use linfa::prelude::*;
use linfa::Float;
use linfa_nn::distance::Distance;
use ndarray_rand::rand::Rng;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug)]
pub enum ProgressType {
    InitDone,
    IterationDone(usize),
}

pub type ProgressCallbackFn = dyn Fn(ProgressType);

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone)]
/// The set of hyperparameters that can be specified for the execution of
/// the [K-means algorithm](crate::KMeans).
pub struct KMeansValidParams<'a, F: Float, R: Rng, D: Distance<F>> {
    /// Number of time the k-means algorithm will be run with different centroid seeds.
    n_runs: usize,
    /// The training is considered complete if the euclidean distance
    /// between the old set of centroids and the new set of centroids
    /// after a training iteration is lower or equal than `tolerance`.
    tolerance: F,
    /// We exit the training loop when the number of training iterations
    /// exceeds `max_n_iterations` even if the `tolerance` convergence
    /// condition has not been met.
    max_n_iterations: u64,
    /// The number of clusters we will be looking for in the training dataset.
    n_clusters: usize,
    /// The initialization strategy used to initialize the centroids.
    init: KMeansInit<F>,
    /// The random number generator
    rng: R,
    /// Distance metric used in the centroid assignment step
    dist_fn: D,
    /// Callback function to notify progress
    pub progress_notify: Option<&'a ProgressCallbackFn>,
}

// impl<'a, F: Float, R: Rng, D: Distance<F>> PartialEq for KMeansValidParams<'a, F, R, D> {
//     fn eq(&self, other: &Self) -> bool {
//         self.n_runs == other.n_runs && self.tolerance == other.tolerance && self.max_n_iterations == other.max_n_iterations && self.n_clusters == other.n_clusters && self.init == other.init && self.rng == other.rng && self.dist_fn == other.dist_fn
//     }
// }

impl<'a, F: Float, R: Rng, D: Distance<F>> std::fmt::Debug for KMeansValidParams<'a, F, R, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KMeansValidParams")
            .field("n_runs", &self.n_runs)
            .field("tolerance", &self.tolerance)
            .field("max_n_iterations", &self.max_n_iterations)
            .field("n_clusters", &self.n_clusters)
            .field("init", &self.init)
            .finish()
    }
}

#[derive(Clone, Debug)]
/// An helper struct used to construct a set of [valid hyperparameters](KMeansParams) for
/// the [K-means algorithm](crate::KMeans) (using the builder pattern).
pub struct KMeansParams<'a, F: Float, R: Rng, D: Distance<F>>(KMeansValidParams<'a, F, R, D>);

impl<'a, F: Float, R: Rng, D: Distance<F>> KMeansParams<'a, F, R, D> {
    /// `new` lets us configure our training algorithm parameters:
    /// * we will be looking for `n_clusters` in the training dataset;
    /// * the training is considered complete if the euclidean distance
    ///   between the old set of centroids and the new set of centroids
    ///   after a training iteration is lower or equal than `tolerance`;
    /// * we exit the training loop when the number of training iterations
    ///   exceeds `max_n_iterations` even if the `tolerance` convergence
    ///   condition has not been met.
    /// * As KMeans convergence depends on centroids initialization
    ///   we run the algorithm `n_runs` times and we keep the best outputs
    ///   in terms of inertia that the ones which minimizes the sum of squared
    ///   euclidean distances to the closest centroid for all observations.
    ///
    /// Defaults are provided if optional parameters are not specified:
    /// * `tolerance = 1e-4`
    /// * `max_n_iterations = 300`
    /// * `n_runs = 10`
    /// * `init = KMeansPlusPlus`
    pub fn new(n_clusters: usize, rng: R, dist_fn: D) -> Self {
        Self(KMeansValidParams {
            n_runs: 10,
            tolerance: F::cast(1e-4),
            max_n_iterations: 300,
            n_clusters,
            init: KMeansInit::KMeansPlusPlus,
            rng,
            dist_fn,
            progress_notify: None,
        })
    }

    /// Change the value of `n_runs`
    pub fn n_runs(mut self, n_runs: usize) -> Self {
        self.0.n_runs = n_runs;
        self
    }

    /// Change the value of `tolerance`
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.0.tolerance = tolerance;
        self
    }

    /// Change the value of `max_n_iterations`
    pub fn max_n_iterations(mut self, max_n_iterations: u64) -> Self {
        self.0.max_n_iterations = max_n_iterations;
        self
    }

    /// Change the value of `init`
    pub fn init_method(mut self, init: KMeansInit<F>) -> Self {
        self.0.init = init;
        self
    }

    /// Change the progress notify callback
    pub fn progress_notify(mut self, progress_notify: &'a ProgressCallbackFn) -> Self {
        self.0.progress_notify = Some(progress_notify);
        self
    }
}

impl<'a, F: Float, R: Rng, D: Distance<F>> ParamGuard for KMeansParams<'a, F, R, D> {
    type Checked = KMeansValidParams<'a, F, R, D>;
    type Error = KMeansParamsError;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        if self.0.n_clusters == 0 {
            Err(KMeansParamsError::NClusters)
        } else if self.0.n_runs == 0 {
            Err(KMeansParamsError::NRuns)
        } else if self.0.tolerance <= F::zero() {
            Err(KMeansParamsError::Tolerance)
        } else if self.0.max_n_iterations == 0 {
            Err(KMeansParamsError::MaxIterations)
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}

impl<'a, F: Float, R: Rng, D: Distance<F>> KMeansValidParams<'a, F, R, D> {
    /// The final results will be the best output of n_runs consecutive runs in terms of inertia.
    pub fn n_runs(&self) -> usize {
        self.n_runs
    }

    /// The training is considered complete if the euclidean distance
    /// between the old set of centroids and the new set of centroids
    /// after a training iteration is lower or equal than `tolerance`.
    pub fn tolerance(&self) -> F {
        self.tolerance
    }

    /// We exit the training loop when the number of training iterations
    /// exceeds `max_n_iterations` even if the `tolerance` convergence
    /// condition has not been met.
    pub fn max_n_iterations(&self) -> u64 {
        self.max_n_iterations
    }

    /// The number of clusters we will be looking for in the training dataset.
    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    /// Cluster initialization strategy
    pub fn init_method(&self) -> &KMeansInit<F> {
        &self.init
    }

    /// Returns the random generator
    pub fn rng(&self) -> &R {
        &self.rng
    }

    /// Returns the distance metric
    pub fn dist_fn(&self) -> &D {
        &self.dist_fn
    }
}

#[cfg(test)]
mod tests {
    use crate::{KMeans, KMeansParams, KMeansParamsError, KMeansValidParams};
    use linfa::ParamGuard;
    use linfa_nn::distance::L2Dist;
    use rand_xoshiro::Xoshiro256Plus;

    #[test]
    fn autotraits() {
        // fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        // has_autotraits::<KMeansParams<f64, Xoshiro256Plus, L2Dist>>();
        // has_autotraits::<KMeansValidParams<f64, Xoshiro256Plus, L2Dist>>();
    }

    #[test]
    fn n_clusters_cannot_be_zero() {
        let res = KMeans::<f32, _>::params(0).check();
        assert!(matches!(res, Err(KMeansParamsError::NClusters)))
    }

    #[test]
    fn tolerance_has_to_positive() {
        let res = KMeans::params(1).tolerance(-1.).check();
        assert!(matches!(res, Err(KMeansParamsError::Tolerance)))
    }

    #[test]
    fn tolerance_cannot_be_zero() {
        let res = KMeans::params(1).tolerance(0.).check();
        assert!(matches!(res, Err(KMeansParamsError::Tolerance)))
    }

    #[test]
    fn max_n_iterations_cannot_be_zero() {
        let res = KMeans::params(1).tolerance(1.).max_n_iterations(0).check();
        assert!(matches!(res, Err(KMeansParamsError::MaxIterations)))
    }

    #[test]
    fn n_runs_cannot_be_zero() {
        let res = KMeans::params(1).tolerance(1.).n_runs(0).check();
        assert!(matches!(res, Err(KMeansParamsError::NRuns)))
    }
}
