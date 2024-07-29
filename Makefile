dev-size:
	clear && du ./target/debug/haru_cmaes -h

prod-size:
	clear && du ./target/release/haru_cmaes -h

ex:
	clear && cargo run --release --example simple_use

clean:
	clear && cargo cache --autoclean && cargo clean

graph-dep:
	# graphviz must be installed: sudo apt install graphviz
	clear && cargo depgraph --all-deps | dot -Tpng > dependencies_graph_of_current_cargo_toml.png

tree:
	clear && cargo tree

deps:
	clear && make tree && make graph-dep

prep:
	clear && cargo clippy && cargo build && cargo fmt

bch:
	clear && cargo bench --bench cmaes

prof:
	clear && cargo run --release --example flamegraph

test:
	clear && cargo test --tests

run:
	clear && cargo run
	
rel:
	clear && cargo run --release

doc:
	clear && cargo doc

pub:
	# cargo login mytoken
	clear && cargo publish
