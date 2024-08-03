use anyhow::Result;
use ndarray::Array2;

pub fn into_f_major(mat: &Array2<f32>) -> Result<Array2<f32>> {
    let shape = mat.dim();
    // println!("");
    // println!("{:+.4?}", &mat);
    let vt: Vec<f32> = Vec::from_iter(mat.t().iter().copied());
    // println!("");
    // println!("Raw tansposed vec...\n{:+.4?}", &vt);
    let f_mat: Array2<f32> = Array2::from_shape_vec(shape, vt).unwrap().t().to_owned();
    // println!("");
    // println!("{:+.4?}", f_mat);
    // println!("");
    Ok(f_mat)
}
