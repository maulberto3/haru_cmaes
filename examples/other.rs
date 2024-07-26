fn main() {
    #[allow(unused_imports)]
    use blas_src;
    use ndarray::prelude::*;
    use ndarray::Array;
    use ndarray_linalg::{assert_close_l2, error, random, Eig, FactorizeInto, Scalar, Solve};

    fn dot() {
        let a = array![[10., 20., 30., 40.,],];
        let b = Array::range(0., 4., 1.); // b = [0., 1., 2., 3, ]
        println!("a shape {:?}", &a.shape());
        println!("b shape {:?}", &b.shape());

        let b = b.to_shape((4, 1)).unwrap(); // reshape b to shape [4, 1]
        println!("b shape after reshape {:?}", &b.shape());

        println!("{}", a.dot(&b)); // [1, 4] x [4, 1] -> [1, 1]
        println!("{}", a.t().dot(&b.t())); // [4, 1] x [1, 4] -> [4, 4]
    }
    dot();

    // Eigen Decomposition
    let a: Array2<f64> = array![
        [1.01, 0.86, 4.60,],
        [3.98, -0.53, 7.04,],
        [3.30, 8.26, 3.89,],
    ];
    let (eigs, vecs) = a.eig().unwrap();
    println!("{:+.4?}", eigs.mapv(|x| x.re));
    println!("{:+.4?}", vecs.mapv(|x| x.re));

    let a = a.map(|v| v.as_c());
    for (&e, vec) in eigs.iter().zip(vecs.axis_iter(Axis(1))) {
        let ev = vec.map(|v| v * e);
        let av = a.dot(&vec);
        assert_close_l2!(&av, &ev, 1e-5);
    }

    // Solve `Ax=b`
    fn solve() -> Result<(), error::LinalgError> {
        let a: Array2<f64> = random((3, 3));
        let b: Array1<f64> = random(3);
        let _x = a.solve(&b)?;
        println!("Done.");
        Ok(())
    }
    solve().unwrap();

    // Solve `Ax=b` for many b with fixed A
    fn factorize() -> Result<(), error::LinalgError> {
        let a: Array2<f64> = random((3, 3));
        let f = a.factorize_into()?; // LU factorize A (A is consumed)
        for _ in 0..10 {
            let b: Array1<f64> = random(3);
            let _x = f.solve_into(b)?; // solve Ax=b using factorized L, U
        }
        println!("Done.");
        Ok(())
    }
    factorize().unwrap();
}
