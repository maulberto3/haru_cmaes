dev-size:
	clear && du ./target/debug/haru_cmaes -h
prod-size:
	clear && du ./target/release/haru_cmaes -h

clean:
	cargo cache --autoclean && cargo clean
lint:
	cargo fmt --check && cargo clippy -- -D warnings
test:
	cargo test --tests
cove:
	cargo tarpaulin --out Html
tree:
	cargo tree
graph-dep:
	cargo depgraph --all-deps | dot -Tpng > dependencies_graph_of_current_cargo_toml.png
deps:
	make tree && make graph-dep
prep:
	cargo machete && cargo build
doct:
	cargo doc
exam:
	cargo run --release --example simple_use
build:
	clear && make clean && make lint && make test && make cove && make deps && make prep && make doct && make exam

benc:
	clear && cargo bench --bench mine
prof:
	clear && cargo run --release --example flamegraph

clif:
	git cliff -o CHANGELOG.md

VERSION := $(shell awk -F ' = ' '/^version/ {gsub(/"/, "", $$2); print $$2}' Cargo.toml)
publ:
	# must have done commits before running the following command
	clear && git diff-index --quiet HEAD || { echo "Uncommitted changes! Commit before publishing."; exit 1; }
	clear && cargo publish && make clif && git push origin master && git tag -a v$(VERSION) -m "Release v$(VERSION)" && git push --tags
	# sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"