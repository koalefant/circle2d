#[cfg(feature = "serde_support")]
pub mod hashmap_as_pairs {
    use rustc_hash::FxHashMap;
    use serde::de::{Deserialize, Deserializer};
    use serde::ser::{Serialize, Serializer};
    pub fn serialize<S, K, V>(map: &FxHashMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        K: Serialize,
        V: Serialize,
    {
        serializer.collect_seq(map)
    }

    pub fn deserialize<'de, D, K, V>(deserializer: D) -> Result<FxHashMap<K, V>, D::Error>
    where
        D: Deserializer<'de>,
        K: Deserialize<'de> + std::cmp::Eq + std::hash::Hash,
        V: Deserialize<'de>,
    {
        let mut map = FxHashMap::default();
        for item in Vec::<(K, V)>::deserialize(deserializer)? {
            map.insert(item.0, item.1);
        }
        Ok(map)
    }
}

#[cfg(feature = "serde_support")]
pub mod btreemap_as_pairs {
    use serde::de::{Deserialize, Deserializer};
    use serde::ser::{Serialize, Serializer};
    use std::collections::BTreeMap;
    pub fn serialize<S, K, V>(map: &BTreeMap<K, V>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        K: Serialize,
        V: Serialize,
    {
        serializer.collect_seq(map)
    }

    pub fn deserialize<'de, D, K, V>(deserializer: D) -> Result<BTreeMap<K, V>, D::Error>
    where
        D: Deserializer<'de>,
        K: Deserialize<'de> + std::cmp::Ord,
        V: Deserialize<'de>,
    {
        let mut map = BTreeMap::new();
        for item in Vec::<(K, V)>::deserialize(deserializer)? {
            map.insert(item.0, item.1);
        }
        Ok(map)
    }
}
