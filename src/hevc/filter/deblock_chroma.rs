//! The chroma half of the deblocking filter, clause 8.7.2.5.5.
//!
//! Chroma is only ever filtered with the short two-tap form — there is no
//! strong filter and no `dE` derivation — so it shares nothing with the luma
//! path but the tables and the segment addressing, both of which stay in the
//! parent module.

use super::super::{FilterInfo, chroma_qp_from_luma};
use super::{BS, Seg, TC_TABLE, clip3, locked};
use crate::hevc::picture::{Picture, Plane, edge};

/// Filters one four-line chroma segment of both chroma planes.
fn segment(pic: &mut Picture, info: &FilterInfo, ver: bool, e: usize, s: usize) {
    let (sw, sh) = (info.sub_w, info.sub_h);
    let (lqx, lqy) = if ver {
        (e * sw, s * sh)
    } else {
        (s * sw, e * sh)
    };
    let (lpx, lpy) = if ver { (lqx - 1, lqy) } else { (lqx, lqy - 1) };
    let qc = info.ctb_of(lqx, lqy);
    let cf = match info.ctb.get(qc) {
        Some(c) => *c,
        None => return,
    };
    if cf.deblock_disabled || !info.may_cross(info.ctb_of(lpx, lpy), qc) {
        return;
    }
    let pl = locked(pic.cu_flags[pic.idx4(lpx, lpy)], info);
    let ql = locked(pic.cu_flags[pic.idx4(lqx, lqy)], info);
    if pl && ql {
        return;
    }
    let qpl = (pic.qp_y[pic.idx4(lqx, lqy)] as i32 + pic.qp_y[pic.idx4(lpx, lpy)] as i32 + 1) >> 1;
    let scale = 1i32 << info.bit_depth_c.saturating_sub(8);
    let max = (1i32 << info.bit_depth_c) - 1;
    for comp in 0..2 {
        let off = i32::from(if comp == 0 {
            cf.cb_qp_offset
        } else {
            cf.cr_qp_offset
        });
        let qpc = chroma_qp_from_luma(qpl + off, info.chroma_array_type);
        let qtc = clip3(
            0,
            53,
            qpc + 2 * (BS - 1) + ((cf.tc_offset_div2 as i32) << 1),
        );
        let tc = TC_TABLE[qtc as usize] * scale;
        let plane: &mut Plane = if comp == 0 { &mut pic.cb } else { &mut pic.cr };
        let seg = Seg::new(plane.stride, ver, e, s, 2);
        let data = &mut plane.data;
        for i in 0..4 {
            let (p1, p0) = (data[seg.at(i, 0)] as i32, data[seg.at(i, 1)] as i32);
            let (q0, q1) = (data[seg.at(i, 2)] as i32, data[seg.at(i, 3)] as i32);
            let d = clip3(-tc, tc, (((q0 - p0) << 2) + p1 - q1 + 4) >> 3);
            if !pl {
                data[seg.at(i, 1)] = clip3(0, max, p0 + d) as u16;
            }
            if !ql {
                data[seg.at(i, 2)] = clip3(0, max, q0 - d) as u16;
            }
        }
    }
}

/// How many strips of eight chroma rows [`strip`] divides the picture into.
pub(super) fn strips(pic: &Picture) -> usize {
    pic.cb.height.div_ceil(8)
}

/// Filters the marked chroma edges of one direction in strip `strip` of eight
/// chroma rows: the vertical edges crossing it, or the horizontal edge on top
/// of it.
pub(super) fn strip(pic: &mut Picture, info: &FilterInfo, ver: bool, strip: usize) {
    if info.chroma_array_type == 0 || info.sub_w == 0 || info.sub_h == 0 {
        return;
    }
    let kind = if ver { edge::VER } else { edge::HOR };
    let (sw, sh) = (info.sub_w, info.sub_h);
    let (cw, ch) = (pic.cb.width, pic.cb.height);
    let (elen, slen) = if ver { (cw, ch) } else { (ch, cw) };
    // Vertical edges: every one, over the strip's rows. Horizontal: the one
    // on top of the strip, over every column.
    let (edges, segments) = if ver {
        (
            (8..elen).step_by(8),
            (8 * strip..slen.min(8 * strip + 8)).step_by(4),
        )
    } else {
        let e = 8 * strip;
        ((e.max(8)..elen.min(e + 1)).step_by(8), (0..slen).step_by(4))
    };
    for e in edges {
        if e + 1 >= elen {
            break;
        }
        for s in segments.clone() {
            if s + 3 >= slen {
                break;
            }
            let (lx, ly) = if ver {
                (e * sw, s * sh)
            } else {
                (s * sw, e * sh)
            };
            let cell = (ly >> 3) * pic.min8_w + (lx >> 3);
            if pic.edges.get(cell).map_or(0u8, |v| *v) & kind != 0 {
                segment(pic, info, ver, e, s);
            }
        }
    }
}
