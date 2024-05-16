use aleo_std_profiler::{end_timer, start_timer};
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use k256::elliptic_curve::sec1;
use num_bigint::BigInt;
use num_bigint::Sign;
use once_cell::sync::Lazy;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use scopeguard::defer;
use serde::Serialize;
use snarkvm_utilities::Write;
use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::AtomicUsize;
use std::time::Instant;
use tempfile::Builder;
use time::format_description::well_known::Rfc3339;
use time::macros::offset;

use crate::console;

fn u64_4_from_slice(x: &[u8]) -> [u64; 4] {
    let array: [u64; 4] = x
        .chunks(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<u64>>()
        .try_into()
        .unwrap();
    array
}

static OUTPUT_DIR: Lazy<PathBuf> = Lazy::new(|| {
    let now = time::OffsetDateTime::now_utc()
        .replace_offset(offset!(+8))
        .format(&Rfc3339)
        .unwrap();
    // let now = "debug".to_string();
    let p = std::path::Path::new(&std::env::var("OUTPUT_DIR").unwrap_or("outputs".to_string()))
        .join(now);

    std::fs::create_dir_all(&p).unwrap();
    p.canonicalize().unwrap()
});

static VERIFY_PLONKY2_COUNT: AtomicUsize = AtomicUsize::new(0);
pub fn build_r1cs_for_verify_plonky2(msgs: Vec<Vec<u8>>) -> Result<()> {
    println!("outdir: {:?}", &*OUTPUT_DIR);
    let tag: String = format!(
        "verifyplonky2-{}-N:{}-LEN:{}",
        VERIFY_PLONKY2_COUNT.load(std::sync::atomic::Ordering::SeqCst),
        msgs.len(),
        msgs[0].len()
    );
    let build_time = start_timer!(|| format!("build_r1cs({})", tag));
    defer! {
        end_timer!(build_time);
    }

    let base_dir = OUTPUT_DIR.join(tag);
    let prove_time = start_timer!(|| "plonky2_evm::hash2::prove_and_aggregate");
    let (tuple, _) = plonky2_evm::hash2::prove_and_aggregate(msgs).unwrap();
    end_timer!(prove_time);

    
    std::fs::create_dir_all(&base_dir).unwrap();
    let ser_time = start_timer!(|| "write to json files");
    std::thread::scope(|s| {
        let encoded = serde_json::to_vec(&tuple.2).unwrap();
        s.spawn(|| std::fs::write(base_dir.join("common_circuit_data.json"), encoded).unwrap());
        let encoded = serde_json::to_vec(&tuple.1).unwrap();
        s.spawn(|| {
            std::fs::write(base_dir.join("verifier_only_circuit_data.json"), encoded).unwrap()
        });
        let encoded = serde_json::to_vec(&tuple.0).unwrap();
        s.spawn(|| {
            std::fs::write(base_dir.join("proof_with_public_inputs.json"), encoded).unwrap()
        });
    });
    end_timer!(ser_time);

    let invoke_time = start_timer!(|| "invoke gnark-plonky2-verifier");
    std::process::Command::new("../gnark-plonky2-verifier/benchmark")
        .args(&[
            "-proof-system",
            "groth16",
            "-plonky2-circuit",
            base_dir.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    end_timer!(invoke_time);

    let construct_r1cs_from_file = start_timer!(|| "construct_r1cs_from_file");
    let ret = super::builder::construct_r1cs_from_file(
        base_dir.join("r1cs.cbor"),
        base_dir.join("assignment.cbor"),
        Some(base_dir.join("lookup.cbor")),
    );
    end_timer!(construct_r1cs_from_file);
    ret
}
// input: 50 * data

// (1) keccak <- input * 50 -> 1 proof , 50 * hash  | 1 * proof with ProofWithPublicInputs<GoldilocksField, PoseidonGoldilocksConfig, 2>, VerifierOnlyCircuitData<PoseidonGoldilocksConfig, 2>, CommonCircuitData<GoldilocksField, 2>
// gnark-verifier <- 3*json
//  ->  r1cs(3*json) -> veruna
// ------
// gnark-ecdsa <-

// output: proof

pub fn build_r1cs_for_verify_ecdsa(
    public_key: &console::ECDSAPublicKey,
    signature: &console::ECDSASignature,
    hash: &[u8; 32],
) -> Result<()> {
    let build_time = start_timer!(|| "build_r1cs_for_verify_ecdsa()");
    defer! {
        end_timer!(build_time);
    }

    // let ret = super::builder::construct_r1cs_from_file(
    //     "../gnark-ecdsa-test/output/r1cs.cbor",
    //     "../gnark-ecdsa-test/output/assignment.cbor",
    //     Some("../gnark-ecdsa-test/output/lookup.cbor"),
    // );

    // return ret;

    let (pk_x, pk_y) = if let sec1::Coordinates::Uncompressed { x, y } =
        public_key.public_key.to_encoded_point(false).coordinates()
    {
        // use plonky2_01::field::secp256k1_base::Secp256K1Base;
        (
            BigInt::from_bytes_be(Sign::Plus, &x.to_vec()),
            BigInt::from_bytes_be(Sign::Plus, &y.to_vec()),
        )
        // (
        // Secp256K1Base(u64_4_from_slice(&x.to_vec())).to_canonical_biguint(),
        // Secp256K1Base(u64_4_from_slice(&y.to_vec())).to_canonical_biguint(),
        // )
    } else {
        unreachable!();
    };

    let (sig_r, sig_s) = {
        // use plonky2_01::field::secp256k1_scalar::Secp256K1Scalar;
        let (r, s) = signature.signature.split_bytes();
        // let r: Vec<u8> = r.to_vec();
        // let s: Vec<u8> = s.to_vec();
        (
            BigInt::from_bytes_be(Sign::Plus, &r.to_vec()),
            BigInt::from_bytes_be(Sign::Plus, &s.to_vec()),
        )
        // (
        //     Secp256K1Scalar(u64_4_from_slice(&r)).to_canonical_biguint(),
        //     Secp256K1Scalar(u64_4_from_slice(&s)).to_canonical_biguint(),
        // )
    };

    let hash = {
        // use plonky2_01::field::secp256k1_scalar::Secp256K1Scalar;
        BigInt::from_bytes_be(Sign::Plus, &hash.to_vec())
        // Secp256K1Scalar(u64_4_from_slice(hash)).to_canonical_biguint()
    };

    let _tmp_dir = Builder::new().prefix("zprize-ecdsa-varuna").tempdir()?;
    let tmp_dir = _tmp_dir.path();

    // go run your_program.go -pk_x "123456789012345678901234567890" -pk_y "987654321098765432109876543210" -sig_r "112233445566778899001122334455" -sig_s "998877665544332211009988776655" -hash "123123123123123123123123123123"
    let output_dir = tmp_dir.join("output");
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("Failed to create output dir at: {output_dir:?}"))?;
    run_external_process(
        Command::new(env::current_dir()?.join("../gnark-ecdsa-test/main"))
            .args(&[
                "-pk_x",
                pk_x.to_str_radix(10).as_str(),
                "-pk_y",
                pk_y.to_str_radix(10).as_str(),
                "-sig_r",
                sig_r.to_str_radix(10).as_str(),
                "-sig_s",
                sig_s.to_str_radix(10).as_str(),
                "-hash",
                hash.to_str_radix(10).as_str(),
            ])
            .current_dir(&tmp_dir),
    )
    .context("Failed to execute gnark-ecdsa-test")?;

    let ret = super::builder::construct_r1cs_from_file(
        output_dir.join("r1cs.cbor"),
        output_dir.join("assignment.cbor"),
        Some(output_dir.join("lookup.cbor")),
    );
    ret
}

fn run_external_process(cmd: &mut Command) -> Result<()> {
    // println!("run cmd: {:?}", cmd);
    let status = cmd
        .status()
        .with_context(|| format!("Failed to execute{:?}", cmd))?;
    if !status.success() {
        bail!("Run process {:?} exited: {}", cmd, status);
    }
    Ok(())
}
