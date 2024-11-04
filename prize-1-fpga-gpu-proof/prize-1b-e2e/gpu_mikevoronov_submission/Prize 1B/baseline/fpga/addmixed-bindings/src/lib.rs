#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(deref_nullptr)]
#![allow(dead_code)]
#![allow(improper_ctypes)]

mod bindings;


use ark_bls12_381::G1Projective;
use ark_bls12_381::Fq;

use crate::bindings::addmixed_op::addmixed_op;
use crate::bindings::addmixed_op::argument_element_t;
use crate::bindings::addmixed_op::result_element_t;

use std::mem;


fn fpga_addmixed(a: G1Projective, b: G1Projective, c: G1Projective)  {

    // data ext
    let c_type = arkworks_to_c_type_arg(a, b);

    //let mut res = G1Projective::default(); 
    let mut res_type = arkworks_to_c_type_res(c);

    //let mut c_type_res = arkworks_to_c_type(G1Projective::default());
    unsafe {
        addmixed_op(
            &c_type as *const argument_element_t,
            &mut res_type as *mut result_element_t,
        );
    }

}



// ========================
// type conversions
// ========================

fn arkworks_to_c_type_arg(a: G1Projective, b: G1Projective) -> argument_element_t {
    argument_element_t {
        x1: a.x.0 .0,
        y1: a.y.0 .0,
        z1: a.z.0 .0,
        x2: b.x.0 .0,
        y2: b.y.0 .0,
    }
}

fn arkworks_to_c_type_res(c: G1Projective) -> result_element_t {
    result_element_t {
        x3: c.x.0 .0,
        y3: c.y.0 .0,
        z3: c.z.0 .0,
    }
}

