//! Inverse transforms (clause 8.6.4).
mod dct;
#[cfg(feature = "bench")]
pub use dct::inverse_transform;
pub use dct::{inverse_transform_into, transform_skip};

mod dequant;
#[cfg(feature = "bench")]
pub use dequant::scale;
pub use dequant::scale_within;

#[cfg(any(test, feature = "bench"))]
mod extent;
