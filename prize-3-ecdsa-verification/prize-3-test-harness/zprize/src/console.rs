use ecdsa::signature::DigestVerifier;
use k256::ecdsa::{Signature, SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use rand::{thread_rng, RngCore};
use sha3::{Digest, Keccak256};
use snarkvm_console_network::prelude::Uniform;

//
// Data structures
// ===============
//

#[derive(Clone)]
pub struct ECDSAPublicKey {
    pub public_key: k256::ecdsa::VerifyingKey,
}

#[derive(Clone)]
pub struct ECDSASignature {
    pub signature: k256::ecdsa::Signature,
}

//
// Sampling functions
// ==================
//
// These are useful as dummy values when we just want to compile the circuit
// (and don't care about the assignment).
//

/// Samples a message of length `size`.
pub fn sample_msg(size: usize) -> Vec<u8> {
    let rng = &mut OsRng::default();
    let mut res = Vec::with_capacity(size);
    for _ in 0..size {
        res.push(u8::rand(rng));
    }
    res
}

/// Samples a public key and signature based on a message
pub fn sample_pubkey_sig(msg: &[u8]) -> (ECDSAPublicKey, ECDSASignature) {
    // generate keypair
    let rng = &mut OsRng::default();
    let secret_key = k256::ecdsa::SigningKey::random(rng);
    let public_key = secret_key.verifying_key().clone();

    // hash bytes
    let mut hasher = sha3::Keccak256::new();
    hasher.update(&msg);

    // sign
    let (signature, _) = secret_key.sign_digest_recoverable(hasher.clone()).unwrap();

    (ECDSAPublicKey { public_key }, ECDSASignature { signature })
}

/// Generates `num` tuples of (public key, msg, signature) with messages of length `msg_len`.
pub fn generate_signatures(msg_len: usize, num: usize) -> Vec<(VerifyingKey, Vec<u8>, Signature)> {
    let mut res = vec![];

    for _ in 0..num {
        // generate keypair
        let secret_key: SigningKey = SigningKey::random(&mut rand::rngs::OsRng);
        let public_key = secret_key.verifying_key();

        // generate msg_len random bytes
        let mut msg = vec![0u8; msg_len];
        thread_rng().fill_bytes(&mut msg);

        // hash msg
        let mut hasher = Keccak256::new();
        hasher.update(&msg);

        // generate signature
        let (signature, _) = secret_key.sign_digest_recoverable(hasher.clone()).unwrap();

        // verify the signature
        assert!(public_key.verify_digest(hasher, &signature).is_ok());

        res.push((public_key.clone(), msg, signature));
    }

    res
}
