dev-size:
	clear && du ./target/debug/haru_cmaes -h

prod-size:
	clear && du ./target/release/haru_cmaes -h

ex:
	#  --features=openblas
	clear && cargo run --example ex1 

clean:
	clear && cargo cache --autoclean && cargo clean

graph-dep:
	# graphviz must be installed: sudo apt install graphviz
	cargo depgraph --all-deps | dot -Tpng > dependencies_graph_of_current_cargo_toml.png

tree:
	clear && cargo tree

deps:
	make tree && make graph-dep

prep:
	clear && cargo fmt && cargo clippy && cargo build

test:
	clear && cargo test --lib

run:
	clear && cargo run
	
rel:
	clear && cargo run --release
