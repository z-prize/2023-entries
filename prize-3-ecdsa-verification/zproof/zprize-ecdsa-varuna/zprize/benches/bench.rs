// Copyright (C) 2019-2023 Aleo Systems Inc.
// This file is part of the snarkVM library.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use zprize::CircuitRunType;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Proof creation");
    group
        .sample_size(10)
        .sampling_mode(criterion::SamplingMode::Flat); // for slow benchmarks

    // let run_type = CircuitRunType::RunKeccakOnly;
    // let run_type = CircuitRunType::RunEcdsaOnly;
    // let run_type = CircuitRunType::RunKeccakAndEcdsa;
    let run_type = CircuitRunType::RunKeccakThenEcdsa;

    // setup
    let urs = zprize::api::setup(1000, 1000, 1000);

    // we generate 50 tuples for each bench
    // tuple = (public key, message, signature)
    let num = 50;

    // 100 bytes
    let msg_len = 100;
    let small_tuples = zprize::console::generate_signatures(msg_len, num);
    let small_circuit_keys = zprize::api::compile(run_type, &urs, num, msg_len);

    // 1,000 bytes
    let msg_len = 1000;
    let medium_tuples = zprize::console::generate_signatures(msg_len, num);
    let medium_circuit_keys = zprize::api::compile(run_type, &urs, num, msg_len);

    // 50,000 bytes
    let msg_len = 50000;
    let large_tuples = zprize::console::generate_signatures(msg_len, num);
    let large_circuit_keys = zprize::api::compile(run_type, &urs, num, msg_len);

    //
    // WARNING
    // =======
    //
    // Do not modify anything above this line.
    // Everything after this line should be fairgame,
    // as long as proofs verify.
    //

    group.bench_function("small message", |b| {
        b.iter(|| {
            // prove all tuples
            zprize::prove_and_verify(
                run_type,
                &urs,
                &small_circuit_keys,
                black_box(&small_tuples),
            );
        })
    });

    group.bench_function("medium message", |b| {
        b.iter(|| {
            // prove all tuples
            zprize::prove_and_verify(
                run_type,
                &urs,
                &medium_circuit_keys,
                black_box(&medium_tuples),
            );
        })
    });

    group.bench_function("large message", |b| {
        b.iter(|| {
            // prove all tuples
            zprize::prove_and_verify(
                run_type,
                &urs,
                &large_circuit_keys,
                black_box(&large_tuples),
            );
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
