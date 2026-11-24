//! Inverse transforms (clause 8.6.4).
mod dct;
pub use dct::{inverse_transform, inverse_transform_into, transform_skip};

mod dequant;
pub use dequant::{scale, scale_within};

mod extent;
