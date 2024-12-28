fn main() {
    #[cfg(feature = "openblas")]
    {
        println!("cargo:rustc-link-lib=openblas");
    }
}