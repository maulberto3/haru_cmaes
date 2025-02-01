pub fn median(mut data: Vec<f32>) -> f32 {
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = data.len();
    if len % 2 == 0 {
        // Average the two middle values
        (data[len / 2 - 1] + data[len / 2]) / 2.0
    } else {
        // Take the middle value
        data[len / 2]
    }
}
