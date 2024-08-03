dev-size:
	clear && du ./target/debug/haru_cmaes -h
prod-size:
	clear && du ./target/release/haru_cmaes -h
tree:
	clear && cargo tree
graph-dep:
	clear && cargo depgraph --all-deps | dot -Tpng > dependencies_graph_of_current_cargo_toml.png
deps:
	clear && make tree && make graph-dep
clean:
	clear && cargo cache --autoclean && cargo clean
doct:
	clear && cargo doc
prep:
	clear && cargo clippy && cargo fmt && cargo build --jobs 2
benc:
	clear && cargo bench --bench cmaes --jobs 2
prof:
	clear && cargo run --release --example flamegraph --jobs 2
exam:
	clear && cargo run --release --example simple_use --jobs 2
test:
	clear && cargo test --tests --jobs 2
run :
	clear && cargo run --jobs 2
rele:
	clear && cargo run --release --jobs 2
# cargo login mytoken
publ:
	clear && cargo publish