use ark_bls12_381::{Bls12_381, Fr, G1Affine};
use ark_ec::PairingEngine;
use ark_ed_on_bls12_381::EdwardsParameters;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_poly_commit::kzg10::UniversalParams;
use ark_poly_commit::sonic_pc::CommitterKey;
use ark_poly_commit::PolynomialCommitment;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::RngCore;
use ark_std::{test_rng, UniformRand};
use merkle_tree::HEIGHT;
use merkle_tree::{MerkleTree, MerkleTreeCircuit};
use plonk_core::commitment::{HomomorphicCommitment, KZG10};
use plonk_core::permutation::Permutation;
use plonk_core::permutation::GPU_POLY_CONTAINER;
use plonk_core::permutation::GPU_POLY_KERNEL;
use plonk_core::permutation::MSM_KERN;
use plonk_core::prelude::{
    verify_proof, Circuit, StandardComposer, VerifierData,
};
use plonk_core::proof_system::gpu_prover::{
    compute_linear_gpu, compute_quotient_gpu, prepare_domain,
    prepare_domain_8n_coset,
};
use plonk_core::proof_system::prover::{DOMAIN, DOMAIN_8N};
use plonk_core::proof_system::Prover;
use plonk_core::proof_system::{ProverKey, VerifierKey};
use plonk_hashing::poseidon::constants::PoseidonConstants;
use plonk_hashing::poseidon::poseidon_ref::NativeSpecRef;

use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::io::{Read, Write};

fn main() {
    let filename = "./data/commiter_key.txt";

    if let Ok(metadata) = fs::metadata(filename) {
        println!("File '{}' exists.", filename);
        let now = std::time::Instant::now();

        // ==============================
        // last we generate the plonk proof
        // ==============================
        {
            // public parameters
            // let size = 1 << (HEIGHT + 9);
            let coeffs_count = 1 << (HEIGHT + 7);

            // proof generation

            println!("main 7,spent:{:?}", now.elapsed());
            let now = std::time::Instant::now();

            // ck pp  vk real_circuit
            type F = Bls12_381;
            let mut commiter_key_bytes = vec![];
            read_from_file(
                "commiter_key.txt".to_string(),
                &mut commiter_key_bytes,
            );
            let ck: CommitterKey<F> = CommitterKey::deserialize_unchecked(
                commiter_key_bytes.as_slice(),
            )
            .unwrap();

            let mut pp_bytes = vec![];
            read_from_file("pp.txt".to_string(), &mut pp_bytes);
            let pp: UniversalParams<F> =
                UniversalParams::deserialize_unchecked(pp_bytes.as_slice())
                    .unwrap();

            type T = ark_bls12_381::Fr;

            let mut pk_bytes = vec![];
            read_from_file("pk.txt".to_string(), &mut pk_bytes);
            let pk: ProverKey<T> =
                ProverKey::deserialize_unchecked(pk_bytes.as_slice()).unwrap();

            // vk.serialize_unchecked(writer)
            type PC = KZG10<F>;
            let mut vk_bytes = vec![];
            read_from_file("vk.txt".to_string(), &mut vk_bytes);
            let vk: VerifierKey<T, PC> =
                VerifierKey::deserialize_unchecked(vk_bytes.as_slice())
                    .unwrap();

            let mut real_circuit_bytes = vec![];
            read_from_file(
                "real_circuit.txt".to_string(),
                &mut real_circuit_bytes,
            );
            let mut real_circuit: MerkleTreeCircuit =
                MerkleTreeCircuit::deserialize_unchecked(
                    real_circuit_bytes.as_slice(),
                )
                .unwrap();

            let mut gpu_context = <KZG10<Bls12_381>>::get_gpu_context(
                &MSM_KERN,
                &ck,
                coeffs_count,
            );
            let kern = &GPU_POLY_KERNEL;
            let gpu_container = &mut GPU_POLY_CONTAINER.lock().unwrap();
            let domain = &DOMAIN;
            let domain_8n = &DOMAIN_8N;
            prepare_domain(kern, gpu_container, domain, true).unwrap();
            prepare_domain(kern, gpu_container, domain_8n, false).unwrap();
            // handle domain_8n coset
            prepare_domain_8n_coset(kern, gpu_container, &pk, domain_8n)
                .unwrap();

            println!("start gen proof, spent:{:?}", now.elapsed());
            let mut prover =
                Prover::<Fr, EdwardsParameters, KZG10<Bls12_381>>::new_2(
                    b"Merkle tree",
                    MSM_KERN.core.get_context(),
                    HEIGHT as u32,
                    Some(&pk),
                    Some(domain),
                    Some(kern),
                    Some(gpu_container),
                );

            //check correct
            for i in 0..200 {
                let proof_now: std::time::Instant = std::time::Instant::now();
                let now = std::time::Instant::now();

                let (proof, pi) = {
                    real_circuit.gadget_2(prover.mut_cs()).unwrap();
                    println!("gadget_2 spent:{:?}", now.elapsed());

                    real_circuit
                        .gen_proof::<KZG10<Bls12_381>, G1Affine>(
                            &mut prover,
                            &ck,
                            &pk,
                            &pp,
                            b"Merkle tree",
                            Some(&mut gpu_context),
                            gpu_container,
                        )
                        .unwrap()
                };

                println!(
                    "The prove generation time is {:?}, i:{}",
                    proof_now.elapsed(),
                    i
                );
                let verifier_data = VerifierData::new(vk.clone(), pi.clone());
                let res = verify_proof::<Fr, EdwardsParameters, KZG10<Bls12_381>>(
                    &pp,
                    verifier_data.key.clone(),
                    &proof,
                    &verifier_data.pi,
                    b"Merkle tree",
                );

                // println!("prove && verify end,spent:{:?}",
                // proof_now.elapsed());

                //println!("public input: {:?}", pi);
                //println!("verifier data: {:?}", verifier_data);

                assert!(res.is_ok());
                println!("proof is verified: {}", res.is_ok());
                println!("");
            }

            println!("check correct");
            println!("");
            // return;
            //warm up
            for _ in 0..10 {
                let (_, _) = {
                    real_circuit.gadget_2(prover.mut_cs()).unwrap();
                    real_circuit
                        .gen_proof::<KZG10<Bls12_381>, G1Affine>(
                            &mut prover,
                            &ck,
                            &pk,
                            &pp,
                            b"Merkle tree",
                            Some(&mut gpu_context),
                            gpu_container,
                        )
                        .unwrap()
                };
            }

            println!("warm up end");
            println!("");

            // run
            let proof_now: std::time::Instant = std::time::Instant::now();
            for _i in 0..10 {
                let (_proof, _pi) = {
                    real_circuit.gadget_2(prover.mut_cs()).unwrap();

                    real_circuit
                        .gen_proof::<KZG10<Bls12_381>, G1Affine>(
                            &mut prover,
                            &ck,
                            &pk,
                            &pp,
                            b"Merkle tree",
                            Some(&mut gpu_context),
                            gpu_container,
                        )
                        .unwrap()
                };
            }

            println!(
                "The prove generation time is {:?}",
                proof_now.elapsed() / 10
            );
        }
    } else {
        println!("File '{}' does not exist.", filename);

        let mut rng = test_rng();
        let now = std::time::Instant::now();

        let param = PoseidonConstants::<Fr>::generate::<3>();
        println!("main 0,spent:{:?}", now.elapsed());
        let now = std::time::Instant::now();

        // ==============================
        // first we build a merkle tree
        // ==============================

        let leaf_nodes = (0..1 << (HEIGHT - 1))
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();
        let tree = MerkleTree::<NativeSpecRef<Fr>>::new_with_leaf_nodes(
            &param,
            &leaf_nodes,
        );
        println!("main 2,spent:{:?}", now.elapsed());
        let now = std::time::Instant::now();

        let index = rng.next_u32() % (1 << (HEIGHT - 1));
        let proof = tree.gen_proof(index as usize);
        let _res = proof.verify(&param, &tree.root());
        println!("main 3,spent:{:?}", now.elapsed());
        let now = std::time::Instant::now();

        // ==============================
        // next we generate the constraints for the tree
        // ==============================

        let mut composer = StandardComposer::<Fr, EdwardsParameters>::new();
        tree.gen_constraints(&mut composer, &param);
        println!("main 3-1,spent:{:?}", now.elapsed());
        let now = std::time::Instant::now();

        composer.check_circuit_satisfied();
        println!("main 4,spent:{:?}", now.elapsed());
        let now = std::time::Instant::now();

        // ==============================
        // last we generate the plonk proof
        // ==============================
        {
            // public parameters
            let size = 1 << (HEIGHT + 9);
            let coeffs_count = 1 << (HEIGHT + 7);

            let pp = KZG10::<Bls12_381>::setup(size, None, &mut rng).unwrap();

            // store end
            println!("main 5,spent:{:?}", now.elapsed());
            let now = std::time::Instant::now();

            let mut dummy_circuit = MerkleTreeCircuit {
                param: param.clone(),
                merkle_tree: tree,
            };

            // proprocessing
            let (pk, (vk, _pi_pos)) =
                dummy_circuit.compile::<KZG10<Bls12_381>>(&pp).unwrap();
            println!("main 6,spent:{:?}", now.elapsed());
            let now = std::time::Instant::now();

            // proof generation
            let leaf_nodes = (0..1 << (HEIGHT - 1))
                .map(|_| Fr::rand(&mut rng))
                .collect::<Vec<_>>();
            let tree = MerkleTree::<NativeSpecRef<Fr>>::new_with_leaf_nodes(
                &param,
                &leaf_nodes,
            );
            let mut real_circuit = MerkleTreeCircuit {
                param: param.clone(),
                merkle_tree: tree,
            };

            println!("main 7,spent:{:?}", now.elapsed());
            let now = std::time::Instant::now();

            let (ck, _) = <KZG10<Bls12_381>>::trim(
                &pp,
                real_circuit.padded_circuit_size(),
                0,
                None,
            )
            .unwrap();
            println!("main 8,spent:{:?}", now.elapsed());
            let now = std::time::Instant::now();

            // ck pp  vk real_circuit
            let mut commiter_key_bytes = vec![];
            ck.serialize_unchecked(&mut commiter_key_bytes).unwrap();
            write_to_file(
                "commiter_key.txt".to_string(),
                commiter_key_bytes.clone(),
            );

            let mut pp_bytes = vec![];
            pp.serialize_unchecked(&mut pp_bytes).unwrap();
            write_to_file("pp.txt".to_string(), pp_bytes.clone());

            type T = ark_bls12_381::Fr;

            //pk
            let mut pk_bytes = vec![];
            pk.serialize_unchecked(&mut pk_bytes).unwrap();
            write_to_file("pk.txt".to_string(), pk_bytes.clone());

            type F = Bls12_381;
            // vk.serialize_unchecked(writer)
            type PC = KZG10<F>;
            let mut vk_bytes = vec![];
            vk.serialize_unchecked(&mut vk_bytes).unwrap();
            write_to_file("vk.txt".to_string(), vk_bytes.clone());

            let mut real_circuit_bytes = vec![];
            real_circuit.serialize(&mut real_circuit_bytes).unwrap();
            write_to_file(
                "real_circuit.txt".to_string(),
                real_circuit_bytes.clone(),
            );

            let kern = &GPU_POLY_KERNEL;
            let gpu_container = &mut GPU_POLY_CONTAINER.lock().unwrap();
            let mut gpu_context = <KZG10<Bls12_381>>::get_gpu_context(
                &MSM_KERN,
                &ck,
                coeffs_count,
            );

            let domain = &DOMAIN;
            let domain_8n = &DOMAIN_8N;
            prepare_domain(kern, gpu_container, domain, true).unwrap();
            prepare_domain(kern, gpu_container, domain_8n, false).unwrap();
            // handle domain_8n coset
            prepare_domain_8n_coset(kern, gpu_container, &pk, domain_8n)
                .unwrap();

            let proof_now = std::time::Instant::now();
            println!("start gen proof");
            let (proof, pi) = {
                let mut prover =
                    Prover::<Fr, EdwardsParameters, KZG10<Bls12_381>>::new_2(
                        b"Merkle tree",
                        MSM_KERN.core.get_context(),
                        HEIGHT as u32,
                        Some(&pk),
                        Some(domain),
                        Some(kern),
                        Some(gpu_container),
                    );
                real_circuit.gadget_2(prover.mut_cs()).unwrap();

                real_circuit
                    .gen_proof::<KZG10<Bls12_381>, G1Affine>(
                        &mut prover,
                        &ck,
                        &pk,
                        &pp,
                        b"Merkle tree",
                        Some(&mut gpu_context),
                        gpu_container,
                    )
                    .unwrap()
            };
            println!("The prove generation time is {:?}", proof_now.elapsed());

            let verifier_data = VerifierData::new(vk, pi.clone());
            let res = verify_proof::<Fr, EdwardsParameters, KZG10<Bls12_381>>(
                &pp,
                verifier_data.key.clone(),
                &proof,
                &verifier_data.pi,
                b"Merkle tree",
            );

            println!("main end,spent:{:?}", now.elapsed());

            //println!("proof: {:?}", proof);
            //println!("public input: {:?}", pi);
            //println!("verifier data: {:?}", verifier_data);
            println!("proof is verified: {}", res.is_ok());
        }
    }
}

fn write_to_file(file_name: String, data: Vec<u8>) {
    let directory_path = "./data";

    // 创建目录
    match fs::create_dir_all(directory_path) {
        Ok(_) => {
            println!("Directory '{}' created successfully.", directory_path)
        }
        Err(err) => println!("Error creating directory: {}", err),
    }

    let file_name = "./data/".to_string() + &file_name;
    println!("filenamek:{}", file_name);

    if let Ok(mut file) = File::create(file_name) {
        if let Err(err) = file.write_all(&data) {
            eprintln!("Failed to write data to file: {}", err);
        }
    } else {
        eprintln!("Failed to create file.");
    };
}

fn read_from_file(file_name: String, mut read_data: &mut Vec<u8>) {
    let file_name = "./data/".to_string() + &file_name;

    if let Ok(mut file) = File::open(file_name) {
        if let Err(err) = file.read_to_end(&mut read_data) {
            eprintln!("Failed to read data from file: {}", err);
        }
    } else {
        eprintln!("Failed to open file.");
    }
}
