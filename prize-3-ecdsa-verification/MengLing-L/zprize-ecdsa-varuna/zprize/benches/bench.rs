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

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Proof creation");
    group
        .sample_size(10)
        .sampling_mode(criterion::SamplingMode::Flat); // for slow benchmarks

    // setup
    let urs = zprize::api::setup(20000000, 20000000, 20000000);

    // we generate 50 tuples for each bench
    // tuple = (public key, message, signature)
    let num = 50;

    // 100 bytes
    let msg_len = 100;
    let small_tuples = zprize::console::generate_signatures(msg_len, num);
    let (small_pk, small_vk) = zprize::api::compile(&urs, msg_len);

    // 1,000 bytes
    let msg_len = 1000;
    let medium_tuples = zprize::console::generate_signatures(msg_len, num);
    let (medium_pk, medium_vk) = zprize::api::compile(&urs, msg_len);

    // 50,000 bytes
    let msg_len = 50000;
    let large_tuples = zprize::console::generate_signatures(msg_len, num);
    let (large_pk, large_vk) = zprize::api::compile(&urs, msg_len);

    //
    // WARNING
    // =======
    //
    // Do not modify anything above this line.
    // Everything after this line should be fairgame,
    // as long as proofs verify.
    //

    // For the small message, we require 2 proofs, and batch 25 signatures per proof
    group.bench_function("small message", |b| {
        b.iter(|| {
            // prove all tuples
            // If your machine cannot support the number of signatures per proof we provide, 
            // or if you have better machine than the one we used, 
            // you can modify the number of batch signatures here
            for small_tuple_chunks in small_tuples.chunks(25){
                zprize::prove_and_verify(
                    &urs,
                    &small_pk,
                    &small_vk,
                    black_box(&small_tuple_chunks.to_vec()),
                );
            }
        })
    });

    // For the medium message, we require 3 proofs, and batch 20, 20, and 10 signatures for each proof, respectively.
    group.bench_function("medium message", |b| {
        b.iter(|| {
            // prove all tuples
            // If your machine cannot support the number of signatures per proof we provide, 
            // or if you have better machine than the one we used, 
            // you can modify the number of batch signatures here
            for medium_tuple_chunks in medium_tuples.chunks(20) {
                zprize::prove_and_verify(
                    &urs,
                    &medium_pk,
                    &medium_vk,
                    black_box(&medium_tuple_chunks.to_vec()),
                );
            }
        })
    });

    // For the large message, we require 17 proofs and batch 3 signatures per proof, except for the last proof, which batches two signatures.
    group.bench_function("large message", |b| {
        b.iter(|| {
            // prove all tuples
            // If your machine cannot support the number of signatures per proof we provide, 
            // or if you have better machine than the one we used, 
            // you can modify the number of batch signatures here
            for large_tuple_chunks in large_tuples.chunks(3) {
                zprize::prove_and_verify(
                    &urs,
                    &large_pk,
                    &large_vk,
                    black_box(&large_tuple_chunks.to_vec()),
                );
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);