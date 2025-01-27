#[cfg(test)]
mod tests {
    use haru_cmaes::ask_tell_use;

    #[test]
    fn test_ask_tell_use() {
        assert!(ask_tell_use::ask_tell_example().is_ok());
    }

    // #[test]
    // fn test_express_use() {
    //     let obj_func = SquareAndSum {
    //         obj_dim: 5,
    //         dir: MinOrMax::Min,
    //     };
    //     assert_eq!(express_use::express_executor(obj_func), true);
    // }
}
