// #[allow(unused_imports)]
// use blas_src;
use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

fn main() {
    let shape = (3, 3);
    let mat: Array2<f32> = Array2::random(shape, Uniform::new(-1., 1.));
    println!("");
    println!("Original data (row-major)\n{:+.4?}", mat);

    let vt: Vec<f32> = Vec::from_iter(mat.t().iter().copied());
    println!("");
    println!("Raw tansposed vec...\n{:+.4?}", &vt);

    let mat3 = Array2::from_shape_vec(shape, vt).unwrap().t().to_owned();
    println!("");
    println!("Original data (column major)\n{:+.4?}", mat3);
    println!("");
}
