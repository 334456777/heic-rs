//! Deblocking filter for intra-coded pictures, ITU-T H.265 clause 8.7.2.
//!
//! Every filtered edge of an intra-only picture has boundary strength 2, so the
//! derivation of clause 8.7.2.4 collapses to the constant [`BS`].

use super::FilterInfo;
use crate::hevc::picture::{Picture, edge, flags};

#[path = "deblock_chroma.rs"]
mod chroma;

/// `beta'` of Table 8-12, indexed by `Q` in `0..=51`.
const BETA_TABLE: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
];

/// `tc'` of Table 8-12, indexed by `Q` in `0..=53`.
const TC_TABLE: [i32; 54] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
    3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24,
];

/// Boundary strength of every filtered edge of an intra-only picture.
const BS: i32 = 2;

/// `Clip3(lo, hi, v)` of clause 5.8, with no panicking path.
#[inline]
pub(super) fn clip3(lo: i32, hi: i32, v: i32) -> i32 {
    v.max(lo).min(hi)
}

/// Where the samples of one four-line segment sit in a plane.
///
/// Sample `j` of line `i` is at `first + i * line + j * along`, where `j`
/// counts from the outermost tap on the P side. Reading an edge of either
/// direction is then the same arithmetic with different steps, so the
/// direction never has to be re-examined per sample.
pub(super) struct Seg {
    first: usize,
    along: usize,
    line: usize,
}

impl Seg {
    /// The segment of the edge at `e`, starting at `s`, whose filter reads
    /// `taps` samples on each side of the edge.
    #[inline]
    pub(super) fn new(stride: usize, ver: bool, e: usize, s: usize, taps: usize) -> Seg {
        if ver {
            Seg {
                first: s * stride + e - taps,
                along: 1,
                line: stride,
            }
        } else {
            Seg {
                first: (e - taps) * stride + s,
                along: stride,
                line: 1,
            }
        }
    }

    /// Plane index of sample `j` of line `i`.
    #[inline]
    pub(super) fn at(&self, i: usize, j: usize) -> usize {
        self.first + i * self.line + j * self.along
    }
}

/// True when loop filtering must leave the samples of this coding unit alone.
#[inline]
fn locked(f: u8, info: &FilterInfo) -> bool {
    f & flags::TQ_BYPASS != 0 || (f & flags::PCM != 0 && info.pcm_loop_filter_disabled)
}

/// The samples of a four-line segment by tap: `t[j][i]` is tap `j` (`p3,p2,p1,p0,q0,q1,q2,q3`) of line `i`.
///
/// Every line of a segment is filtered with the same decisions, so laid out this way each step of the filter is the
/// same operation on four values side by side, which the compiler turns into vector instructions.
type Taps = [[i32; 4]; 8];

/// Filters the four lines of a luma segment, clause 8.7.2.5.7.
fn filter_lines(t: &Taps, de: u8, dep: bool, deq: bool, tc: i32, max: i32) -> Taps {
    let mut o = *t;
    let [p3, p2, p1, p0, q0, q1, q2, q3] = t;
    if de == 2 {
        for i in 0..4 {
            let (p3, p2, p1, p0) = (p3[i], p2[i], p1[i], p0[i]);
            let (q0, q1, q2, q3) = (q0[i], q1[i], q2[i], q3[i]);
            o[3][i] = clip3(
                p0 - 2 * tc,
                p0 + 2 * tc,
                (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3,
            );
            o[2][i] = clip3(p1 - 2 * tc, p1 + 2 * tc, (p2 + p1 + p0 + q0 + 2) >> 2);
            o[1][i] = clip3(
                p2 - 2 * tc,
                p2 + 2 * tc,
                (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3,
            );
            o[4][i] = clip3(
                q0 - 2 * tc,
                q0 + 2 * tc,
                (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3,
            );
            o[5][i] = clip3(q1 - 2 * tc, q1 + 2 * tc, (p0 + q0 + q1 + q2 + 2) >> 2);
            o[6][i] = clip3(
                q2 - 2 * tc,
                q2 + 2 * tc,
                (p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3,
            );
        }
    } else if de == 1 {
        // Whether a line is filtered depends on its own samples: each line works out its new values and keeps
        // them or its old ones, rather than branching.
        for i in 0..4 {
            let (p2, p1, p0) = (p2[i], p1[i], p0[i]);
            let (q0, q1, q2) = (q0[i], q1[i], q2[i]);
            let raw = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
            let on = raw.abs() < tc * 10;
            let d = clip3(-tc, tc, raw);
            o[3][i] = if on { clip3(0, max, p0 + d) } else { p0 };
            o[4][i] = if on { clip3(0, max, q0 - d) } else { q0 };
            if dep {
                let dv = clip3(-(tc >> 1), tc >> 1, (((p2 + p0 + 1) >> 1) - p1 + d) >> 1);
                o[2][i] = if on { clip3(0, max, p1 + dv) } else { p1 };
            }
            if deq {
                let dv = clip3(-(tc >> 1), tc >> 1, (((q2 + q0 + 1) >> 1) - q1 - d) >> 1);
                o[5][i] = if on { clip3(0, max, q1 + dv) } else { q1 };
            }
        }
    }
    o
}

/// Filters one four-line luma segment of the edge at `e` starting at `s`.
fn luma_segment(pic: &mut Picture, info: &FilterInfo, ver: bool, e: usize, s: usize) {
    let (px, py) = if ver { (e - 1, s) } else { (s, e - 1) };
    let (qx, qy) = if ver { (e, s) } else { (s, e) };
    let qc = info.ctb_of(qx, qy);
    let cf = match info.ctb.get(qc) {
        Some(c) => *c,
        None => return,
    };
    if cf.deblock_disabled || !info.may_cross(info.ctb_of(px, py), qc) {
        return;
    }
    let pl = locked(pic.cu_flags[pic.idx4(px, py)], info);
    let ql = locked(pic.cu_flags[pic.idx4(qx, qy)], info);
    if pl && ql {
        return;
    }
    let qpl = (pic.qp_y[pic.idx4(qx, qy)] as i32 + pic.qp_y[pic.idx4(px, py)] as i32 + 1) >> 1;
    let scale = 1i32 << info.bit_depth_y.saturating_sub(8);
    let beta = BETA_TABLE[clip3(0, 51, qpl + ((cf.beta_offset_div2 as i32) << 1)) as usize] * scale;
    let qtc = clip3(
        0,
        53,
        qpl + 2 * (BS - 1) + ((cf.tc_offset_div2 as i32) << 1),
    );
    let tc = TC_TABLE[qtc as usize] * scale;
    let seg = Seg::new(pic.y.stride, ver, e, s, 4);
    let data = &mut pic.y.data;
    // A line of a vertical edge's segment is its eight taps side by side, and a tap of a horizontal edge's is its
    // four lines side by side: each is read as one slice.
    let mut t: Taps = [[0; 4]; 8];
    if ver {
        for i in 0..4 {
            for (tap, &v) in t.iter_mut().zip(&data[seg.at(i, 0)..][..8]) {
                tap[i] = i32::from(v);
            }
        }
    } else {
        for (j, tap) in t.iter_mut().enumerate() {
            for (v, &sample) in tap.iter_mut().zip(&data[seg.at(0, j)..][..4]) {
                *v = i32::from(sample);
            }
        }
    }
    let dp = |i: usize| (t[1][i] - 2 * t[2][i] + t[3][i]).abs();
    let dq = |i: usize| (t[6][i] - 2 * t[5][i] + t[4][i]).abs();
    let (dp0, dq0) = (dp(0), dq(0));
    let (dp3, dq3) = (dp(3), dq(3));
    let (dpq0, dpq3) = (dp0 + dq0, dp3 + dq3);
    if dpq0 + dpq3 >= beta {
        return;
    }
    let sam = |i: usize, dpq: i32| {
        2 * dpq < (beta >> 2)
            && (t[0][i] - t[3][i]).abs() + (t[4][i] - t[7][i]).abs() < (beta >> 3)
            && (t[3][i] - t[4][i]).abs() < ((5 * tc + 1) >> 1)
    };
    let de = if sam(0, dpq0) && sam(3, dpq3) {
        2u8
    } else {
        1u8
    };
    let thr = (beta + (beta >> 1)) >> 3;
    let (dep, deq) = (dp0 + dp3 < thr, dq0 + dq3 < thr);
    let max = (1i32 << info.bit_depth_y) - 1;
    let o = filter_lines(&t, de, dep, deq, tc, max);
    // Only p2..q2 can change, and a locked side keeps its samples.
    let (lo, hi) = (if pl { 4 } else { 1 }, if ql { 4 } else { 7 });
    if ver {
        for i in 0..4 {
            let line = &mut data[seg.at(i, 0)..][..8];
            for (v, tap) in line[lo..hi].iter_mut().zip(&o[lo..hi]) {
                *v = tap[i] as u16;
            }
        }
    } else {
        for (j, tap) in o.iter().enumerate().take(hi).skip(lo) {
            for (v, &n) in data[seg.at(0, j)..][..4].iter_mut().zip(tap) {
                *v = n as u16;
            }
        }
    }
}

/// Filters every marked luma edge of one direction over the whole picture.
fn luma_dir(pic: &mut Picture, info: &FilterInfo, ver: bool) {
    let kind = if ver { edge::VER } else { edge::HOR };
    let (w, h) = (pic.y.width, pic.y.height);
    let (elen, slen) = if ver { (w, h) } else { (h, w) };
    for by in 0..pic.min8_h {
        for bx in 0..pic.min8_w {
            if pic.edges[by * pic.min8_w + bx] & kind == 0 {
                continue;
            }
            let (e, s0) = if ver {
                (bx * 8, by * 8)
            } else {
                (by * 8, bx * 8)
            };
            if e == 0 || e + 3 >= elen {
                continue;
            }
            for seg in 0..2 {
                let s = s0 + seg * 4;
                if s + 3 < slen {
                    luma_segment(pic, info, ver, e, s);
                }
            }
        }
    }
}

/// Applies the in-loop deblocking filter of clause 8.7.2 to `pic` in place.
///
/// All vertical edges of the picture are filtered before any horizontal edge,
/// so the horizontal pass reads samples the vertical pass already modified.
pub fn deblock(pic: &mut Picture, info: &FilterInfo) {
    luma_dir(pic, info, true);
    chroma::dir(pic, info, true);
    luma_dir(pic, info, false);
    chroma::dir(pic, info, false);
}

#[cfg(test)]
#[path = "deblock_tests.rs"]
mod tests;
