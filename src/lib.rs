mod sdf;
mod serialization;
mod spatial_hash;
mod world;
#[allow(unused_imports)]
use sdf::*;
#[cfg(feature = "serde_support")]
use serialization::*;
use spatial_hash::*;
pub use world::*;

#[cfg(feature = "fixed_point")]
mod fixed_math;
#[cfg(feature = "fixed_point")]
use fixed_math::*;
#[cfg(feature = "fixed_point")]
mod fixed_trig;
#[cfg(feature = "fixed_point")]
use fixed_trig::*;

/// bit mask that can be either used together with World::set_collision_shape_flags
/// to mark which bodies can collide, or mark bodies with arbitrary flags
pub type ShapeFlags = u32;

/// can be used to identify owner of the body from outside, may be redefined locally
pub type BodyOwner = u64;

/// this can be locally overriden, for example, to use fixed-point math or doubles
#[cfg(feature = "fixed_point")]
pub type Real = fixed_math::Fixed;
#[cfg(not(feature = "fixed_point"))]
pub type Real = f32;

/// used to construct internal constants in a way that survives redefinition of real
#[cfg(not(feature = "fixed_point"))]
pub fn reali(v: i32) -> Real {
    v as Real
}
#[cfg(not(feature = "fixed_point"))]
pub fn realf(f: f32) -> Real {
    f
}
#[cfg(feature = "fixed_point")]
pub fn reali(v: i32) -> Real {
    fixed_math::Fixed::from_i32(v)
}
#[cfg(feature = "fixed_point")]
pub fn realf(v: f32) -> Real {
    fixed_math::Fixed::from_f32(v)
}

/// this can be locally overriden, for example, to u    se fixed-point math or doubles
#[cfg(feature = "fixed_point")]
pub type Real2 = fixed_math::Fixed2;
#[cfg(not(feature = "fixed_point"))]
pub type Real2 = glam::Vec2;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

pub fn hash_slot_map<
    K: slotmap::Key + std::hash::Hash,
    V: std::hash::Hash + slotmap::Slottable,
    H: std::hash::Hasher,
>(
    map: &slotmap::SlotMap<K, V>,
    h: &mut H,
) {
    map.iter().for_each(|(k, v)| {
        k.hash(h);
        v.hash(h);
    });
}

pub fn hash_secondary_map<
    K: slotmap::Key + std::hash::Hash,
    V: std::hash::Hash,
    H: std::hash::Hasher,
>(
    map: &slotmap::SecondaryMap<K, V>,
    h: &mut H,
) {
    map.iter().for_each(|(k, v)| {
        k.hash(h);
        v.hash(h);
    });
}
