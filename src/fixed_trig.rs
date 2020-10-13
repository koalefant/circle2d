//
// Contains a fragment of XMunkki/FixPointCS library distributed under MIT license
//
use crate::fixed_math::Fixed64;

pub trait FixedTrig {
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn atan2(self, x: Self) -> Self;
}

pub fn fixed_pi() -> Fixed64 {
    Fixed64::from_bits(PI)
}

impl FixedTrig for Fixed64 {
    fn sin(self) -> Self {
        let z: i32 = mul_int_long_low(RCP_HALF_PI, self.to_bits());
        Self::from_bits((unit_sin(z) as i64) << 2)
    }
    fn cos(self) -> Self {
        Self::from_bits(self.to_bits() + PI_HALF).sin()
    }
    fn atan2(self, x: Self) -> Self {
        // https://www.dsprelated.com/showarticle/1052.php
        let x: i64 = x.to_bits();
        let y: i64 = self.to_bits();
        if x == 0 {
            if y > 0 {
                return Self::from_bits(PI_HALF);
            }
            if y < 0 {
                return Self::from_bits(-PI_HALF);
            }
            panic!("invalid atan2 arguments");
        }

        // these round negative numbers slightly
        let nx: i64 = x ^ (x >> 63);
        let ny: i64 = y ^ (y >> 63);
        let neg_mask: i64 = (x ^ y) >> 63;

        if nx >= ny {
            let k: i32 = atan2_div(ny, nx);
            let z: i32 = atan_poly_5lut8(k);
            let angle: i64 = neg_mask ^ ((z as i64) << 2);
            if x > 0 {
                return Self::from_bits(angle);
            }
            if y >= 0 {
                return Self::from_bits(angle + PI);
            }
            Self::from_bits(angle - PI)
        } else {
            let k: i32 = atan2_div(nx, ny);
            let z: i32 = atan_poly_5lut8(k);
            let angle: i64 = neg_mask ^ ((z as i64) << 2);
            Self::from_bits(if y > 0 { PI_HALF } else { -PI_HALF } - angle)
        }
    }
}

fn sin_poly4(a: i32) -> i32 {
    let y = qmul30(a, 162679); // 0.000151506641710145430212560273580165931825591912723771559939880958777921352896251494433561036087921925941339032487946104446
    let y = qmul30(a, y + -5018587); // -0.0046739239118693360423625115440933405485555388758012378155538229669555462190128366781129325889847935291248353457031014355
    let y = qmul30(a, y + 85566362); // 0.0796898846149471415166275702814129714699583291426024010731469442497475447581642697337742897122044339073717901878121832219
    let y = qmul30(a, y + -693598342); // -0.645963794139684570135799310650651238951748485457327220679639722739088328461814215309859665984340413045934902046607019536
    y + 1686629713 // 1.57079632679489661923132169163975144209852010327780228586210672049751840856976653075976474782503285074174660817200999164
}

fn unit_sin(bits: i32) -> i32 {
    let mut z = bits;
    if (z ^ (z << 1)) < 0 {
        z = ((1 as i32) << 31).wrapping_sub(z);
    }

    // Now z is in range [-1, 1].
    let one: i32 = 1 << 30;
    assert!((z >= -one) && (z <= one));

    // Polynomial approximation.
    let zz: i32 = qmul30(z, z);
    let res: i32 = qmul30(sin_poly4(zz), z);

    // Return s2.30 value.
    res
}

fn nlz(mut x: u64) -> i32 {
    let mut n: i32 = 0;
    if x <= 0x00000000FFFFFFFF {
        n = n + 32;
        x = x << 32;
    }
    if x <= 0x0000FFFFFFFFFFFF {
        n = n + 16;
        x = x << 16;
    }
    if x <= 0x00FFFFFFFFFFFFFF {
        n = n + 8;
        x = x << 8;
    }
    if x <= 0x0FFFFFFFFFFFFFFF {
        n = n + 4;
        x = x << 4;
    }
    if x <= 0x3FFFFFFFFFFFFFFF {
        n = n + 2;
        x = x << 2;
    }
    if x <= 0x7FFFFFFFFFFFFFFF {
        n = n + 1;
    }
    if x == 0 {
        return 64;
    }
    n
}

fn qmul30(a: i32, b: i32) -> i32 {
    (a as i64 * b as i64 >> 30) as i32
}

const RCP_POLY_4LUT8_TABLE: [i32; 40] = [
    796773553,
    -1045765287,
    1072588028,
    -1073726795,
    1073741824,
    456453183,
    -884378041,
    1042385791,
    -1071088216,
    1073651788,
    276544830,
    -708646126,
    977216564,
    -1060211779,
    1072962711,
    175386455,
    -559044324,
    893798171,
    -1039424537,
    1071009496,
    115547530,
    -440524957,
    805500803,
    -1010097984,
    1067345574,
    78614874,
    -348853503,
    720007233,
    -974591889,
    1061804940,
    54982413,
    -278348465,
    641021491,
    -935211003,
    1054431901,
    39383664,
    -223994590,
    569927473,
    -893840914,
    1045395281,
];

const RCP_HALF_PI: i32 = 683565276; // 1.0 / (4.0 * 0.5 * Math.PI);  // the 4.0 factor converts directly to s2.30
const PI: i64 = 13493037705; //(FP_LONG)(Math.PI * 65536.0) << 16;
                             //const PI2: i64 = 26986075409;
const PI_HALF: i64 = 6746518852;

// Precision: 24.07 bits
fn rcp_poly_4lut8(a: i32) -> i32 {
    let offset = ((a >> 27) * 5) as usize;
    let y = qmul30(a, RCP_POLY_4LUT8_TABLE[offset + 0]);
    let y = qmul30(a, y + RCP_POLY_4LUT8_TABLE[offset + 1]);
    let y = qmul30(a, y + RCP_POLY_4LUT8_TABLE[offset + 2]);
    let y = qmul30(a, y + RCP_POLY_4LUT8_TABLE[offset + 3]);
    y + RCP_POLY_4LUT8_TABLE[offset + 4]
}

fn atan2_div(y: i64, x: i64) -> i32 {
    assert!(y >= 0 && x > 0 && x >= y);

    // Normalize input into [1.0, 2.0( range (convert to s2.30).
    let one: i32 = 1 << 30;
    let half: i32 = 1 << 29;
    let offset: i32 = 31 - nlz(x as i64 as u64);
    let n: i32 = ((if offset >= 0 { x >> offset } else { x << -offset }) >> 2) as i32;
    let k: i32 = n - one;

    // Polynomial approximation of reciprocal.
    let oox: i32 = rcp_poly_4lut8(k);
    assert!(oox >= half && oox <= one);

    // Apply exponent and multiply.
    let yr: i64 = if offset >= 0 { y >> offset } else { y << -offset };
    qmul30((yr >> 2) as i32, oox)
}

const ATAN_POLY_5LUT8_TABLE: [i32; 54] = [
    204464916, 1544566, -357994250, 1395, 1073741820, 0, 119369854, 56362968, -372884915, 2107694, 1073588633, 4534,
    10771151, 190921163, -440520632, 19339556, 1071365339, 120610, -64491917, 329189978, -542756389, 57373179,
    1064246365, 656900, -89925028, 390367074, -601765924, 85907899, 1057328034, 1329793, -80805750, 360696628,
    -563142238, 60762238, 1065515580, 263159, -58345538, 276259197, -435975641, -35140679, 1101731779, -5215389,
    -36116738, 179244146, -266417331, -183483381, 1166696761, -16608596, 0, 0, 0, 0, 0, 843314857, // Atan(1.0)
];

// Precision: 28.06 bits
fn atan_poly_5lut8(a: i32) -> i32 {
    let offset = ((a >> 27) * 6) as usize;
    let y = qmul30(a, ATAN_POLY_5LUT8_TABLE[offset + 0]);
    let y = qmul30(a, y + ATAN_POLY_5LUT8_TABLE[offset + 1]);
    let y = qmul30(a, y + ATAN_POLY_5LUT8_TABLE[offset + 2]);
    let y = qmul30(a, y + ATAN_POLY_5LUT8_TABLE[offset + 3]);
    let y = qmul30(a, y + ATAN_POLY_5LUT8_TABLE[offset + 4]);
    y + ATAN_POLY_5LUT8_TABLE[offset + 5]
}

fn mul_int_long_low(a: i32, b: i64) -> i32 {
    assert!(a >= 0);
    let bi: i32 = (b >> 32) as i32;
    let bf: i64 = b & 0xffffffff;
    ((((a as i64 * bf) as u64) >> 32) as i32).wrapping_add(a.wrapping_mul(bi))
}
