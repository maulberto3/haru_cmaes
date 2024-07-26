
#[cfg(test)]
mod tests {
    use haru_cmaes::simple_use;

    #[test]
    fn end_to_end_test() {
        // Assuming simple_use::example() returns a Result
        assert!(simple_use::example().is_ok());
    }
}
