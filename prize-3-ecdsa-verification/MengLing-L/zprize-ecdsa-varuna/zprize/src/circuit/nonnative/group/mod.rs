use super::*;

pub mod add;
pub use add::*;
pub mod double;
pub use double::*;
pub mod mul;
pub use mul::*;

/// Given the point P on secp256k1 curve.
/// it's x,y corrdinates are in the q_256 field.
/// The challenge is how to verify point P is on secp256k1 curve in native field q_253.
///
/// To prove one point is on secp256k1 curve,
/// The corrdinates of the point should satisfy:
/// 1. x^3 + 7 = y^2 mod q_256
/// 2. x, y are in the q_256 field
///
/// In other words, we should ensure that
/// x_2 = x * x mod q_256
/// x_3 = x_2 * x mod q_256
/// y_2 = y * y mod q_256
/// x_3_add_7 = x_3 + 7 mod q_256
/// x_3_add_7 = y_2 mod q_256
///
/// Therefore, we can precompute x_2, x_3, x_3_add_7, y_2 over q_256,
/// and view the above conditions as the non-native mulitiplication
/// or non-native linear combination.
/// Please refer ../basic/mul.rs and ../basic/linear.rs for details
///
#[derive(Clone, Debug)]
pub struct CircuitNonNativePoint {
    pub(crate) x: Vec<F>,
    pub(crate) y: Vec<F>,
    pub(crate) x_2: CircuitNonNativeMul,
    pub(crate) x_3: CircuitNonNativeMul,
    pub(crate) y_2: CircuitNonNativeMul,

    pub(crate) x_3_add_7: CircuitNonNativeLinearComb,

    pub(crate) noninfinity: B,
}

impl Inject for CircuitNonNativePoint {
    type Primitive = ConsoleNonNativePoint<Testnet3>;

    fn new(mode: Mode, console_point: Self::Primitive) -> Self {
        Self {
            x: console_point
                .x
                .0
                .iter()
                .map(|b| F::new(mode, *b))
                .collect_vec(),
            y: console_point
                .y
                .0
                .iter()
                .map(|b| F::new(mode, *b))
                .collect_vec(),
            x_2: CircuitNonNativeMul::new(mode, console_point.x_2),
            x_3: CircuitNonNativeMul::new(mode, console_point.x_3),
            y_2: CircuitNonNativeMul::new(mode, console_point.y_2),
            x_3_add_7: CircuitNonNativeLinearComb::new(mode, console_point.x_3_add_7),
            noninfinity: B::new(mode, console_point.noninfinity),
        }
    }
}

impl Eject for CircuitNonNativePoint {
    type Primitive = ConsoleNonNativePoint<Testnet3>;

    fn eject_mode(&self) -> Mode {
        self.x.eject_mode()
    }

    fn eject_value(&self) -> Self::Primitive {
        ConsoleNonNativePoint {
            x: Polynomial(self.x.eject_value()),
            y: Polynomial(self.y.eject_value()),
            x_2: self.x_2.eject_value(),
            x_3: self.x_3.eject_value(),
            y_2: self.y_2.eject_value(),
            x_3_add_7: self.x_3_add_7.eject_value(),
            x_: todo!(),
            y_: todo!(),
            noninfinity: self.noninfinity.eject_value(),
        }
    }
}

impl NonNativePoint for CircuitNonNativePoint {
    type Size = F;
    /// Verify a point on secp256k1 curve in circuit.
    /// It includes two main steps:
    /// 1. prove x_2, x_3, y_2, x_3_add_7 are well-form over secp256k1 base field.
    /// 2. Check y_2 == x_3_add_7
    ///
    /// Note: If the point is the point at infinity,
    ///         x, y, x_2, x_3, y_2, x_3_add_7 should be zero;
    fn on_curve(&self) {
        CHUNK_SIZE.with(|chunk_size| {
            self.x_2.poly_mul_wellform(&chunk_size, &self.x, &self.x);
            self.x_3
                .poly_mul_wellform(&chunk_size, &self.x_2.c, &self.x);
            self.y_2.poly_mul_wellform(&chunk_size, &self.y, &self.y);
            NONNATIVE_ZERO.with(|zeros| {
                NONNATIVE_SEVEN.with(|seven| {
                    self.x_3_add_7.poly_linear_combination_wellform(
                        &chunk_size,
                        &vec![
                            (Signed::ADD, self.x_3.c.clone()),
                            (
                                Signed::ADD,
                                Vec::<F>::ternary(&self.noninfinity, seven, zeros),
                            ),
                        ],
                    );
                });

                //check y^2 = x^3 + 7 over secp256 base field
                self.x_3_add_7
                    .value
                    .poly_eq(&self.noninfinity, &self.y_2.c, zeros);
            });
        });
    }
}
