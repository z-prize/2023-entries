# Write-up of zproof team 


## How to run

### Build the project

1. Install golang toolchains, since some of our dependency are written in go

    You may also want to set `GOPROXY` if your network is restricted.
    ```sh
    go env -w GOPROXY=https://goproxy.cn,direct
    ```

3. Install rust toolchains

    ```sh
    # For compatibility we recommend using nightly-2024-01-30
    rustup default nightly-2024-01-30
    ```

3. Install the [`just`](https://github.com/casey/just) tool

4. Clone this project and setup

    ```sh
    git clone --recursive git@github.com:zproof/zprize-ecdsa-varuna.git
    cd zprize-ecdsa-varuna
    just setup
    just build-gnark
    ```

5. Download blobs
   
    Before the first time you generate proof with this project, it would be better to download some of Aleo's blobs in advance. Those files are required by snarkVM, and they will be saved at `~/.aleo`.

    ```sh
    cargo test --package zprize --lib -- tests::download_tons_of_blobs --exact --nocapture
    ```

    Note this is a one-time operation, and you only need to call this once.


### Run the benchmark

Run the bench with:

```sh
cargo bench
```

We have enabled "profiler" feature of `aleo-std-profiler` crate, so that we can get detailed time during the proving process. 


It's worth mentioning that in the benchmarks for each messages length, we generate proof data separately for keccak and ecdsa (This behavior is controlled by `let run_type = CircuitRunType::RunKeccakAndEcdsa`). Therefore, we will call `prove_batch()` twice and will have two proofs. To achieve this, we made some modifications to `zprize::api::compile()`, although we noted the comment in benches/bench.rs advising against modifying parts of the code there.

The time taken to generate a proof (i.e. time used by `prove_batch()`) can be obtained by checking the output of programs for fields similar to `End:     Varuna::Prover 1375.194s`.


