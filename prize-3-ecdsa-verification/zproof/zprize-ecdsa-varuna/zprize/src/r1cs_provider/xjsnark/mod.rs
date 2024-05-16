use std::process::Command;

use anyhow::bail;
use anyhow::Result;
use tempfile::Builder;

pub fn build_r1cs_for_verify_ecdsa(
    msg: &[u8],
    public_key_bytes: &[u8],
    signature: &[u8],
) -> Result<()> {
    let _tmp_dir = Builder::new().prefix("zprize-ecdsa-varuna").tempdir()?;
    let tmp_dir = _tmp_dir.path();

    // 1. call xjsnark to generate .arith and .in
    // Example:
    // java -cp ../zprize_demo/out/production/zprize_demo:../xjsnark/languages/xjsnark/runtime/lib/xjsnark_backend.jar xjsnark.verifyEcdsa.VerifyEcdsa da21f88697f084394d6ce7d5be83f8090ed4abcb464d785db01b8a2234554f69ad29fccf6db0b33f1af2bab4305cc8c7d828f94236435e8b0452c386a259b71f1a7dc30b3c822d00d138b6f5f536cf34d2079292a69274b518602470c74d23ca1893e1fc ef61ca30d74824605da1dd2017d42e7fb2993dd9c193a2680854af5d8b827caf44 69334c873f4b53127c701e331219678243180410a749491016249fe2e6024525668b83604f2125edcd5c9adf54e44dd197a04252ef20563847d3a462a34b4c3f
    let output = Command::new("/usr/lib/jvm/java-8-openjdk/bin/java")
        .args([
            "-cp",
            "../zprize_demo/out/production/zprize_demo/:../xjsnark/languages/xjsnark/runtime/lib/xjsnark_backend.jar",
            "xjsnark.verifyEcdsa.VerifyEcdsa",
            hex::encode(msg).as_str(),
            hex::encode(public_key_bytes).as_str(),
            hex::encode(signature).as_str()
        ])
        .current_dir(tmp_dir)
        .output()?;
    if !output.status.success() {
        bail!("failed to generate circuit with xjsnark");
    }

    // 2. call jsnark(libsnark) to generate r1cs
    // Example:
    // ../jsnark/libsnark/build/libsnark/jsnark_interface/arith_to_r1cs /tmp/VerifyEcdsa.arith /tmp/VerifyEcdsa_Sample_Run1.in
    let output = Command::new(
        "../jsnark/libsnark/build/libsnark/jsnark_interface/arith_to_r1cs",
    )
    .args(["./VerifyEcdsa.arith", "./VerifyEcdsa_Sample_Run1.in"])
    .current_dir(tmp_dir)
    .output()?;
    if !output.status.success() {
        bail!("failed to generate r1cs and assignment json files");
    }

    // 3. read r1cs and .in, then parse it to R1CS in snarkvm
    // super::builder::construct_r1cs_from_file(
    //     tmp_dir.join("r1cs.json"),
    //     tmp_dir.join("assignment.json"),
    //     Option::<&str>::None,
    // )
    todo!()
}
