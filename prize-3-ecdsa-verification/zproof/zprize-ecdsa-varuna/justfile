setup:
    git submodule update --init --recursive
    go env -w GO111MODULE=on
    cd gnark-ecdsa-test && go mod download
    cd gnark-plonky2-verifier && go mod download
    cd zprize && cargo check

build-gnark:
    cd gnark-ecdsa-test && go build ./main.go
    cd gnark-plonky2-verifier && go build ./benchmark.go

test: build-gnark
    cd zprize && cargo test --release