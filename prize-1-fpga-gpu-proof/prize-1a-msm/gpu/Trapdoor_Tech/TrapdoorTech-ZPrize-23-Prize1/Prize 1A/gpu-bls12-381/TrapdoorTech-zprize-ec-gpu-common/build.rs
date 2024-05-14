// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use core::panic;
use std::env;

fn feature_check() {
    let curves = ["bls12_377", "bls12_381"];
    let curves_as_features: Vec<String> = (0..curves.len())
        .map(|i| format!("CARGO_FEATURE_{}", curves[i].to_uppercase()))
        .collect();

    let mut curve_counter = 0;
    for curve_feature in curves_as_features.iter() {
        curve_counter += env::var(&curve_feature).is_ok() as i32;
    }

    match curve_counter {
        0 => panic!("Can't run without a curve being specified, please select one with --features=<curve>. Available options are\n{:#?}\n", curves),
        2.. => panic!("Multiple curves are not supported, please select only one."),
        _ => (),
    };
}

fn main() {
    feature_check();
}
