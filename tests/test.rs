#[cfg(test)]
mod tests {
    use haru_cmaes::ask_tell_use;

    #[test]
    fn end_to_end_test() {
        assert!(ask_tell_use::ask_tell_example().is_ok());
    }
}
