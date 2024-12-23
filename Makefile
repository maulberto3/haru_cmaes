###
dev-size:
	clear && du ./target/debug/libharu_cmaes.rlib -h
prod-size:
	clear && du ./target/release/libharu_cmaes.rlib -h
###
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
###
benc:
	clear && cargo bench --bench mine
prof:
	clear && cargo run --release --example flamegraph
###
VERSION := $(shell awk -F ' = ' '/^version/ {gsub(/"/, "", $$2); print $$2}' Cargo.toml)
clif:
	# Generate the changelog and commit it in the same step
	git cliff -o CHANGELOG.md
	git add .
	git commit -m "Update changelog for v$(VERSION)"
	git push origin master

publ:
	# Check for uncommitted changes
	clear && git diff-index --quiet HEAD || { echo "Uncommitted changes! Commit before publishing."; exit 1; }
	# Perform the publish and then update changelog
	clear && make clif && git tag -a v$(VERSION) -m "Release v$(VERSION)" && git push --tags && cargo publish
	# Optional: Clean cache after publishing (commented out)
	# sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"