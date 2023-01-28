use criterion::{
    black_box, criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion,
    PlotConfiguration,
};
use linfa::benchmarks::config;
use linfa::prelude::*;
use linfa::DatasetBase;
use linfa_clustering::{IncrKMeansError, KMeans, KMeansInit};
use linfa_datasets::generate;
use linfa_nn::distance::Distance;
use linfa_nn::distance::EMD;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform};
use rand_xoshiro::Xoshiro256Plus;

// 170us for a 50 dim distance compare
// 34us for a 10 dim distance compare
fn emd(c: &mut Criterion) {
    let mut rng = Xoshiro256Plus::seed_from_u64(40);
    let cluster_sizes = [(100, 4), (400, 10), (3000, 10), (9000, 10), (9000, 50)];
    let n_features = 3;

    let mut benchmark = c.benchmark_group("emd");
    config::set_default_benchmark_configs(&mut benchmark);
    benchmark.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for &(cluster_size, n_clusters) in &cluster_sizes {
        let rng = &mut rng;
        let centroids =
            Array2::random_using((n_clusters, n_features), Uniform::new(0., 1.), rng);
        let dataset = generate::blobs(cluster_size, &centroids, rng);

        benchmark.bench_function(
            BenchmarkId::new("find_farest", format!("{}x{}",cluster_size, n_clusters )),
            |bencher| {
                bencher.iter(|| {
                    let first_row = dataset.row(0);
                    let mut closest_index = 0;
                    let mut max_distance = 0.0;
                    
                    let iterator = dataset.rows().into_iter();
                    for (row_index, row) in iterator.enumerate() {
                        let distance = EMD.distance(row.view(), first_row.view());
                        if distance > max_distance {
                            closest_index = row_index;
                            max_distance = distance;
                        }
                    }
                    (closest_index, max_distance)
                });
            },
        );
    }

    benchmark.finish();
}

criterion_group!(benches, emd);
criterion_main!(benches);
