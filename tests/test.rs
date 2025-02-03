// #[cfg(test)]
// mod tests {
//     use haru_cmaes::state::CmaesStateLogic;
//     use haru_cmaes::{fitness::MinOrMax, objectives::SquareAndSum};

//     #[test]
//     fn test_ask_tell_use() {
//         assert!(ask_tell::ask_tell_example().is_ok());
//     }

//     #[test]
//     fn test_express_use() {
//         let obj_func = SquareAndSum {
//             obj_dim: 5,
//             dir: MinOrMax::Min,
//         };
//         let state = express_use::express_executor(obj_func);
//         let (_, best_y_fit) = state.get_best().unwrap();
//         let value = best_y_fit[0];
//         println!("{:?}", value);

//         assert!(value.is_finite());
//     }
// }
