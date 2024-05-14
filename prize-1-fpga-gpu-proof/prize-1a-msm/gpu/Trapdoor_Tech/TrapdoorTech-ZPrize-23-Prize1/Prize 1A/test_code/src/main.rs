use ark_ff::FpParameters;
use clap::Parser;

mod sample_msm;

/// ...
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Curve id
    #[arg(short, long, default_value_t = 377)]
    curve: usize,

    /// Degree of the polynomial
    #[arg(short, long, default_value_t = 16777216)]
    degree: usize,

    /// Seed
    #[arg(short, long, default_value_t = 0)]
    seed: u64,

    /// Output folder name
    #[arg(short, long, default_value_t = String::from("data"))]
    output_dir: String,
}

fn main() {
    env_logger::init();

    let args = Args::parse();
    print_info(&args);
    match args.curve {
        377 => {
            sample_msm::sample_msm_377::<ark_bls12_377::g1::Parameters, _>(&args);
        },
        381 => sample_msm::sample_msm_381::<ark_bls12_381::g1::Parameters, _>(&args),
        _ => unreachable!(),
    }
}

fn print_info(args: &Args) {
    println!("======================================================");
    println!("To print more info, use: RUST_LOG=\"info\" cargo run ...");
    println!("======================================================");

    log::info!("curve id:       {}", args.curve);
    log::info!("poly degree:    {}", args.degree);
    log::info!("seed:           {}", args.seed);
    log::info!("output file:    {}", args.output_dir);
    log::info!("modulus:        {}", get_modulus(args.curve));
}

/// Print out the modulus for the ring.
/// This is the scalar field of the curve.
fn get_modulus(curve_id: usize) -> String {
    match curve_id {
        377 => format!(
            "0x{}",
            <ark_bls12_377::FqParameters as FpParameters>::MODULUS
        )
        .to_string(),

        381 => format!(
            "0x{}",
            <ark_bls12_381::FqParameters as FpParameters>::MODULUS
        )
        .to_string(),

        _ => {
            log::error!("do not support curve {}", curve_id);
            unreachable!()
        }
    }
}
