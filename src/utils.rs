use anyhow::Result;
use ndarray::Array2;
use std::{fs::File, io::Read};

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

pub fn get_memory_usage() -> Result<usize> {
    let mut s = String::new();
    File::open("/proc/self/statm")?.read_to_string(&mut s)?;
    let fields: Vec<&str> = s.split_whitespace().collect();
    Ok(fields[1].parse::<usize>().unwrap() * 4096 / 1000000) // Resident Set Size in bytes
}

pub fn format_number(num: usize) -> String {
    let num_str = num.to_string();
    let mut result = String::new();
    let mut count = 0;

    for c in num_str.chars().rev() {
        if count == 3 {
            result.push(',');
            count = 0;
        }
        result.push(c);
        count += 1;
    }

    result.chars().rev().collect()
}
