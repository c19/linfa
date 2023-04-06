use linfa::Float;
use ndarray::{ArrayView, Dimension, Array, Zip};

pub fn subtract<F: Float, D: Dimension>(a: &ArrayView<F, D>, b: &ArrayView<F, D>) -> Array<F, D> {
    let mut result = Array::zeros(a.raw_dim());
    Zip::from(&mut result)
        .and(a)
        .and(b)
        .for_each(|r, &x, &y| *r = y - x);
    result
}

pub fn emd<F: Float, D: Dimension>(seq1: &ArrayView<F, D>, seq2: &ArrayView<F, D>) -> F {
    let delta: Array<F, D> = subtract(seq1 , seq2);
    let mut delta = delta.into_shape(seq1.len()).unwrap();
    let mut cost = F::zero();
    for i in 0..delta.len() {
        if delta[i] != F::zero() {
            for j in 1..delta.len() - i {
                // 向两侧寻找可以填平的空档
                if i >= j {
                    let k = i - j;
                    // 左边
                    if delta[i] * delta[k] < F::zero() {
                        let res = delta[i] + delta[k];
                        if delta[i].abs() <= delta[k].abs() {
                            cost += delta[i].abs() * F::from(j).unwrap();
                            delta[[i]] = F::zero();
                            delta[[k]] = res;
                            break;
                        } else {
                            cost += delta[k].abs() * F::from(j).unwrap();
                            delta[[i]] = res;
                            delta[[k]] = F::zero();
                        }
                    }
                }
                let k = i + j;
                // 右边
                if k < delta.len() && delta[i] * delta[k] < F::zero() {
                    let res = delta[i] + delta[k];
                    if delta[i].abs() <= delta[k].abs() {
                        cost += delta[i].abs() * F::from(j).unwrap();
                        delta[[i]] = F::zero();
                        delta[[k]] = res;
                        break;
                    } else {
                        cost += delta[k].abs() * F::from(j).unwrap();
                        delta[[i]] = res;
                        delta[[k]] = F::zero();
                    }
                }
            }
        }
    }
    cost
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn case_0() {
        let seq1 = array![0.1, 0.3, 0.0, 0.0, 0.6];
        let seq2 = array![0.1, 0.7, 0.0, 0.2, 0.0];
        let result = emd(&seq1.view(), &seq2.view());
        assert_eq!(result, 1.4);
    }

    #[test]
    fn case_1() {
        let seq1 = array![0.0, 0.8, 0.0, 0.0, 0.0, 0.1, 0.1];
        let seq2 = array![0.1, 0.1, 0.0, 0.7, 0.0, 0.0, 0.1];
        let result = emd(&seq1.view(), &seq2.view());
        assert_eq!(result, 1.5);
    }

    #[test]
    fn case_2() {
        let seq1 = array![2.0, 1.0, 0.0, 0.0, 3.0, 0.0, 4.0];
        let seq2 = array![0.0, 5.0, 3.0, 0.0, 2.0, 0.0, 0.0];
        let result = emd(&seq1.view(), &seq2.view());
        assert_eq!(result, 22.0);
    }

    #[test]
    fn case_3() {
        let seq1 = array![2.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let seq2 = array![0.0, 2.0, 1.0, 0.0, 3.0, 0.0, 0.0];
        let result = emd(&seq1.view(), &seq2.view());
        assert_eq!(result, 3.0);
    }

    #[test]
    fn case_4() {
        let seq1 = array![2.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let seq2 = array![0.0, 2.0, 1.0, 0.0, 0.0, 3.0, 0.0];
        let result = emd(&seq1.view(), &seq2.view());
        assert_eq!(result, 6.0);
    }

    #[test]
    fn case_5() {
        let seq1 = array![2.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let seq2 = array![0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0];
        let result = emd(&seq1.view(), &seq2.view());
        assert_eq!(result, 12.0);
    }

    #[test]
    fn case_6() {
        let seq1 = array![0.2, 0.1, 0.0, 0.0, 0.3, 0.0, 0.0];
        let seq2 = array![0.0, 0.2, 0.1, 0.0, 0.0, 0.3, 0.0];
        let result = emd(&seq1.view(), &seq2.view());
        assert!(result - 0.6 < 10e-10);
    }
}
