//
// Contains a fragment of XMunkki/FixPointCS library distributed under MIT license
//
use glam::{vec2, Vec2};
#[cfg(feature = "serde")]
use serde_derive::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Hash, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub struct Fixed64(i64);
impl std::fmt::Display for Fixed64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.*}", f.precision().unwrap_or(3), self.to_f64())
    }
}
impl std::fmt::Debug for Fixed64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.*}", f.precision().unwrap_or(3), self.to_f64())
    }
}

pub type Fixed = Fixed64;

impl Fixed64 {
    pub const MAX: Fixed64 = Fixed64(i64::MAX);
    pub const MIN: Fixed64 = Fixed64(i64::MIN);

    pub fn from_bits(v: i64) -> Self {
        Self(v as i64)
    }
    pub fn from_i32(v: i32) -> Self {
        Self((v as i64) << 32)
    }
    pub fn from_f32(v: f32) -> Self {
        Self((v * (0x100000000 as i64) as f32) as i64)
    }
    pub fn to_i32(self) -> i32 {
        (self.0 >> 32) as i32
    }
    pub fn to_f32(self) -> f32 {
        (self.0 as f32) / (0x100000000 as i64 as f32)
    }
    pub fn to_f64(self) -> f64 {
        (self.0 as f64) / (0x100000000 as i64 as f64)
    }
    pub fn to_bits(self) -> i64 {
        self.0 as i64
    }

    pub fn min(self, rhs: Fixed64) -> Self {
        Self(self.0.min(rhs.0))
    }
    pub fn max(self, rhs: Fixed64) -> Self {
        Self(self.0.max(rhs.0))
    }
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }
    pub fn ceil(self) -> Self {
        Self((self.0 + 0xffffffff) & ((0xffffffff00000000 as u64) as i64))
    }
    pub fn floor(self) -> Self {
        Self(self.0 & ((0xffffffff00000000 as u64) as i64))
    }
    pub fn round(self) -> Self {
        Self((self.0 + 0x7fffffff) & ((0xffffffff00000000 as u64) as i64))
    }
    pub fn signum(self) -> Self {
        if self.0 < 0 {
            Self(-1)
        } else {
            if self.0 > 0 {
                Self(1)
            } else {
                Self(0)
            }
        }
    }
    pub fn zero() -> Self {
        Self(0)
    }

    pub fn max_value() -> Self {
        Self(i64::max_value())
    }
    pub fn min_value() -> Self {
        Self(i64::min_value())
    }

    #[inline]
    fn div_precise(arg_a: i64, arg_b: i64) -> i64 {
        // From http://www.hackersdelight.org/hdcodetxt/divlu.c.txt
        let sign_dif = arg_a ^ arg_b;

        let b: u64 = 0x100000000; // Number base (32 bits)
        let abs_arg_a: u64 = if arg_a < 0 { -arg_a } else { arg_a } as u64;
        let u1: u64 = (abs_arg_a >> 32) as u64;
        let u0: u64 = (abs_arg_a << 32) as u64;
        let mut v = if arg_b < 0 { -arg_b } else { arg_b } as u64;

        // Overflow?
        if u1 >= v {
            //rem = 0;
            return 0x7fffffffffffffff;
        }

        // Shift amount for norm
        let s = v.leading_zeros() as i32; // 0 <= s <= 63
        v = v << s; // Normalize the divisor
        let vn1: u64 = v >> 32; // Break the divisor into two 32-bit digits
        let vn0: u64 = v & 0xffffffff;

        let un32: u64 = (u1 << s) | (u0 >> (64 - s)) & (((-s) as i64 >> 63) as u64);
        let un10: u64 = u0 << s; // Shift dividend left

        let un1: u64 = un10 >> 32; // Break the right half of dividend into two digits
        let un0: u64 = un10 & 0xffffffff;

        // Compute the first quotient digit, q1
        let mut q1: u64 = un32 / vn1;
        let mut rhat: u64 = un32.wrapping_sub(q1.wrapping_mul(vn1));
        loop {
            if (q1 >= b) || ((q1.wrapping_mul(vn0)) > (b.wrapping_mul(rhat).wrapping_add(un1))) {
                q1 = q1 - 1;
                rhat = rhat + vn1;
            } else {
                break;
            }
            if rhat >= b {
                break;
            }
        }

        let un21: u64 = un32
            .wrapping_mul(b)
            .wrapping_add(un1)
            .wrapping_sub(q1.wrapping_mul(v)); // Multiply and subtract

        // Compute the second quotient digit, q0
        let mut q0: u64 = un21 / vn1;
        rhat = un21.wrapping_sub(q0.wrapping_mul(vn1));
        loop {
            if (q0 >= b) || ((q0.wrapping_mul(vn0)) > (b.wrapping_mul(rhat).wrapping_add(un0))) {
                q0 = q0.wrapping_sub(1);
                rhat = rhat.wrapping_add(vn1);
            } else {
                break;
            }
            if rhat >= b {
                break;
            }
        }

        // Calculate the remainder
        // FP_ULONG r = (un21 * b + un0 - q0 * v) >> s;
        // rem = (FP_LONG)r;

        let ret: u64 = q1.wrapping_mul(b).wrapping_add(q0);
        if sign_dif < 0 {
            -(ret as i64)
        } else {
            ret as i64
        }
    }

    #[inline]
    fn rem(a: i64, b: i64) -> i64 {
        let di = a / b;
        a - (di * b)
    }
}

impl std::fmt::Display for Fixed2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({:.*}, {:.*})",
            f.precision().unwrap_or(3),
            self.x.to_f64(),
            f.precision().unwrap_or(3),
            self.y.to_f64()
        )
    }
}
impl std::fmt::Debug for Fixed2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({:.*}, {:.*})",
            f.precision().unwrap_or(3),
            self.x.to_f64(),
            f.precision().unwrap_or(3),
            self.y.to_f64()
        )
    }
}
impl std::ops::Neg for Fixed64 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Fixed64(-self.0)
    }
}
impl std::ops::Mul for Fixed64 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let ai = self.0 >> 32;
        let af = self.0 & 0xffffffff;
        let bi = rhs.0 >> 32;
        let bf = rhs.0 & 0xffffffff;
        Self(
            ((((af.wrapping_mul(bf)) as u64) >> 32) as i64)
                .wrapping_add(ai.wrapping_mul(rhs.0))
                .wrapping_add(af.wrapping_mul(bi)),
        )
    }
}
impl std::ops::Div for Fixed64 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(Self::div_precise(self.0, rhs.0))
    }
}
impl std::ops::Rem for Fixed64 {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        Self(Self::rem(self.0, rhs.0))
    }
}
impl std::ops::Add for Fixed64 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}
impl std::ops::Sub for Fixed64 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}
impl std::ops::AddAssign for Fixed64 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Fixed64) {
        *self = *self + rhs;
    }
}
impl std::ops::SubAssign for Fixed64 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Fixed64) {
        *self = *self - rhs;
    }
}
impl std::ops::MulAssign for Fixed64 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Fixed64) {
        *self = *self * rhs;
    }
}
impl std::ops::DivAssign for Fixed64 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Fixed64) {
        *self = *self / rhs;
    }
}

impl From<f32> for Fixed64 {
    fn from(v: f32) -> Fixed64 {
        Fixed64::from_f32(v)
    }
}

impl Default for Fixed64 {
    fn default() -> Self {
        Self(0)
    }
}

pub trait Ortho {
    fn ortho(&self) -> Self;
}

pub trait FixedSqrt {
    fn sqrt(self) -> Self;
}
impl FixedSqrt for Fixed64 {
    #[inline(always)]
    fn sqrt(self) -> Self {
        // Adapted from https://github.com/chmike/fpsqrt
        let a = self.0;
        if a <= 0 {
            assert!(a >= 0);
            return Self(0);
        }

        let mut r: u64 = a as u64;
        let mut b: u64 = 0x4000000000000000;
        let mut q: u64 = 0;
        while b > 0x40 {
            let t = q + b;
            if r >= t {
                r -= t;
                q = t + b;
            }
            r <<= 1;
            b >>= 1;
        }
        q >>= 16;
        Self(q as i64)
    }
}

#[derive(Copy, Clone, Hash, Ord, Eq)]
#[cfg_attr(feature = "save", derive(Serialize, Deserialize))]
pub struct Fixed2 {
    pub x: Fixed,
    pub y: Fixed,
}

impl Fixed2 {
    #[inline(always)]
    pub fn new(x: Fixed, y: Fixed) -> Self {
        Self { x, y }
    }
    pub fn x(&self) -> Fixed {
        self.x
    }
    pub fn y(&self) -> Fixed {
        self.y
    }
    pub fn set_x(&mut self, x: Fixed) {
        self.x = x;
    }
    pub fn set_y(&mut self, y: Fixed) {
        self.y = y;
    }
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> Fixed {
        self.x * rhs.x + self.y * rhs.y
    }
    pub fn normalize(self) -> Self {
        let len = self.length();
        assert!(len.abs() > Fixed64(16));
        Fixed2::new(self.x / len, self.y / len)
    }
    pub fn normalize_or(self, default: Fixed2) -> Self {
        let len = self.length();
        if len.abs() > Fixed64(16) {
            Fixed2::new(self.x / len, self.y / len)
        } else {
            default
        }
    }
    #[inline(always)]
    pub fn length_squared(self) -> Fixed {
        self.dot(self)
    }
    #[inline(always)]
    pub fn length(self) -> Fixed {
        let len_squared = self.dot(self);
        if len_squared.0 < 0 {
            panic!(
                "negative dot product result: {} for vector {:?}",
                len_squared, self
            );
        }
        len_squared.sqrt()
    }
    pub fn manhattan_length(self) -> Fixed {
        self.x.abs() + self.y.abs()
    }
    pub fn abs(self) -> Fixed2 {
        Fixed2::new(self.x.abs(), self.y.abs())
    }
    pub fn ceil(self) -> Fixed2 {
        Fixed2::new(self.x.ceil(), self.y.ceil())
    }
    pub fn floor(self) -> Fixed2 {
        Fixed2::new(self.x.floor(), self.y.floor())
    }
    pub fn zero() -> Fixed2 {
        Fixed2::new(fixi(0), fixi(0))
    }
    pub fn max(self, rhs: Self) -> Fixed2 {
        Fixed2::new(self.x.max(rhs.x), self.y.max(rhs.y))
    }
    pub fn min(self, rhs: Self) -> Fixed2 {
        Fixed2::new(self.x.min(rhs.x), self.y.min(rhs.y))
    }
    pub fn max_element(self) -> Fixed {
        self.x.max(self.y)
    }
    pub fn min_element(self) -> Fixed {
        self.x.min(self.y)
    }
    pub fn to_float(self) -> Vec2 {
        vec2(self.x.to_f32(), self.y.to_f32())
    }
    pub fn to_i32(self) -> [i32; 2] {
        [self.x.to_i32(), self.y.to_i32()]
    }
}

impl Into<(i32, i32)> for Fixed2 {
    fn into(self) -> (i32, i32) {
        (self.x.to_i32(), self.y.to_i32())
    }
}
impl Into<Vec2> for Fixed2 {
    fn into(self) -> Vec2 {
        Vec2::new(self.x.to_f32(), self.y.to_f32())
    }
}

impl From<Vec2> for Fixed2 {
    fn from(v: Vec2) -> Fixed2 {
        Fixed2::new(Fixed::from_f32(v.x()), Fixed::from_f32(v.y()))
    }
}

impl From<Fixed64> for f32 {
    fn from(v: Fixed64) -> f32 {
        v.to_f32()
    }
}

impl std::cmp::PartialOrd for Fixed2 {
    fn partial_cmp(&self, other: &Fixed2) -> Option<std::cmp::Ordering> {
        let a = self.x.cmp(&other.x);
        if a != std::cmp::Ordering::Equal {
            return Some(a);
        }
        Some(self.y.cmp(&other.y))
    }
}
impl Ortho for Vec2 {
    fn ortho(&self) -> Self {
        Vec2::new(-self.y(), self.x())
    }
}
impl Ortho for Fixed2 {
    fn ortho(&self) -> Self {
        Fixed2::new(-self.y, self.x)
    }
}
impl std::cmp::PartialEq for Fixed2 {
    fn eq(&self, rhs: &Self) -> bool {
        self.x == rhs.x && self.y == rhs.y
    }
    fn ne(&self, rhs: &Self) -> bool {
        self.x != rhs.x || self.y != rhs.y
    }
}
impl std::ops::Neg for Fixed2 {
    type Output = Fixed2;
    #[inline(always)]
    fn neg(self) -> Self {
        Fixed2 {
            x: -self.x,
            y: -self.y,
        }
    }
}
impl std::ops::Add for Fixed2 {
    type Output = Fixed2;
    #[inline(always)]
    fn add(self, rhs: Fixed2) -> Fixed2 {
        Fixed2 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}
impl std::ops::Sub for Fixed2 {
    type Output = Fixed2;
    #[inline(always)]
    fn sub(self, rhs: Fixed2) -> Fixed2 {
        Fixed2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}
impl std::ops::Mul for Fixed2 {
    type Output = Fixed2;
    #[inline(always)]
    fn mul(self, rhs: Fixed2) -> Fixed2 {
        Fixed2 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
        }
    }
}
impl std::ops::Mul<Fixed2> for Fixed {
    type Output = Fixed2;
    #[inline(always)]
    fn mul(self, rhs: Fixed2) -> Fixed2 {
        Fixed2 {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}
impl std::ops::Div for Fixed2 {
    type Output = Fixed2;
    #[inline(always)]
    fn div(self, rhs: Fixed2) -> Fixed2 {
        Fixed2 {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
        }
    }
}
impl std::ops::Div<Fixed> for Fixed2 {
    type Output = Fixed2;
    #[inline(always)]
    fn div(self, rhs: Fixed) -> Fixed2 {
        Fixed2 {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}
impl std::ops::Mul<Fixed> for Fixed2 {
    type Output = Fixed2;
    #[inline(always)]
    fn mul(self, rhs: Fixed) -> Fixed2 {
        Fixed2 {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}
impl std::ops::AddAssign for Fixed2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Fixed2) {
        self.x += rhs.x;
        self.y += rhs.y
    }
}
impl std::ops::SubAssign for Fixed2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Fixed2) {
        self.x -= rhs.x;
        self.y -= rhs.y
    }
}
impl std::ops::MulAssign for Fixed2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Fixed2) {
        self.x *= rhs.x;
        self.y *= rhs.y
    }
}
impl std::ops::MulAssign<Fixed> for Fixed2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Fixed) {
        self.x *= rhs;
        self.y *= rhs
    }
}
impl Default for Fixed2 {
    fn default() -> Fixed2 {
        fix2i(0, 0)
    }
}

impl From<Fixed2> for [f32; 2] {
    fn from(v: Fixed2) -> Self {
        [v.x.to_f32(), v.y.to_f32()]
    }
}

pub trait FixedCopySign {
    fn copysign(self, rhs: Self) -> Self;
}
impl FixedCopySign for Fixed {
    fn copysign(self, rhs: Self) -> Self {
        if self.signum() != rhs.signum() {
            self * fixi(-1)
        } else {
            self
        }
    }
}

pub trait FixedLerp {
    fn lerp_f(self, b: Self, f: Fixed) -> Self;
}
impl FixedLerp for Fixed {
    fn lerp_f(self, b: Self, f: Fixed) -> Self {
        self * (fixi(1) - f) + b * f
    }
}
impl FixedLerp for Fixed2 {
    fn lerp_f(self, b: Self, f: Fixed) -> Self {
        self * (fixi(1) - f) + b * f
    }
}
impl FixedLerp for [u8; 3] {
    fn lerp_f(self, b: Self, f: Fixed) -> Self {
        [
            (fixi(self[0] as _))
                .lerp_f(fixi(b[0] as _), f)
                .to_i32()
                .max(0)
                .min(255) as u8,
            (fixi(self[1] as _))
                .lerp_f(fixi(b[1] as _), f)
                .to_i32()
                .max(0)
                .min(255) as u8,
            (fixi(self[2] as _))
                .lerp_f(fixi(b[2] as _), f)
                .to_i32()
                .max(0)
                .min(255) as u8,
        ]
    }
}
impl FixedLerp for [u8; 4] {
    fn lerp_f(self, b: Self, f: Fixed) -> Self {
        [
            (fixi(self[0] as _))
                .lerp_f(fixi(b[0] as _), f)
                .to_i32()
                .max(0)
                .min(255) as u8,
            (fixi(self[1] as _))
                .lerp_f(fixi(b[1] as _), f)
                .to_i32()
                .max(0)
                .min(255) as u8,
            (fixi(self[2] as _))
                .lerp_f(fixi(b[2] as _), f)
                .to_i32()
                .max(0)
                .min(255) as u8,
            (fixi(self[3] as _))
                .lerp_f(fixi(b[3] as _), f)
                .to_i32()
                .max(0)
                .min(255) as u8,
        ]
    }
}

#[allow(dead_code)]
#[inline(always)]
pub fn fix(v: f32) -> Fixed {
    Fixed::from_f32(v)
}
#[inline(always)]
pub fn fixi(v: i32) -> Fixed {
    Fixed::from_i32(v)
}
#[inline(always)]
pub fn fix2(x: Fixed, y: Fixed) -> Fixed2 {
    Fixed2::new(x, y)
}
#[allow(dead_code)]
#[inline(always)]
pub fn fix2f(x: f32, y: f32) -> Fixed2 {
    Fixed2::new(fix(x), fix(y))
}
#[inline(always)]
pub fn fix2i(x: i32, y: i32) -> Fixed2 {
    Fixed2::new(Fixed::from_i32(x), Fixed::from_i32(y))
}
#[allow(dead_code)]
#[inline(always)]
pub fn fix2v(v: Vec2) -> Fixed2 {
    Fixed2::new(Fixed::from_f32(v.x()), Fixed::from_f32(v.y()))
}
