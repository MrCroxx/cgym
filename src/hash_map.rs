use crossbeam_epoch::{self as epoch, Guard};
use std::{collections::hash_map::RandomState, sync::Arc};

use crate::{
    code::{HashBuilder, Key, Value},
    list_map::ListMapInner,
};

#[derive(Debug)]
struct HashMapInner<K, V, S = RandomState>
where
    K: Key,
    V: Value,
    S: HashBuilder,
{
    bits: u8,
    slots: Vec<ListMapInner<K, V>>,
    hasher: S,
}

impl<K, V, S> HashMapInner<K, V, S>
where
    K: Key,
    V: Value,
    S: HashBuilder,
{
    fn new_with_hasher(bits: u8, hasher: S) -> Self {
        let mut slots = Vec::with_capacity(1 << bits);
        for _ in 0..(1 << bits) {
            slots.push(ListMapInner::new());
        }
        Self {
            bits,
            slots,
            hasher,
        }
    }

    fn insert(&self, key: K, value: V, guard: &Guard) -> bool {
        let h = self.hasher.hash_one(&key);
        let slot = (h % (1 << self.bits)) as usize;
        self.slots[slot].insert(key, value, guard)
    }

    fn remove(&self, key: K, guard: &Guard) -> bool {
        let h = self.hasher.hash_one(&key);
        let slot = (h % (1 << self.bits)) as usize;
        self.slots[slot].remove(key, guard)
    }

    fn get<'g>(&self, key: &'g K, guard: &'g Guard) -> Option<&'g V> {
        let h = self.hasher.hash_one(key);
        let slot = (h % (1 << self.bits)) as usize;
        self.slots[slot].get(key, guard)
    }
}

#[derive(Clone, Debug)]
pub struct HashMap<K, V, S = RandomState>
where
    K: Key,
    V: Value,
    S: HashBuilder,
{
    inner: Arc<HashMapInner<K, V, S>>,
}

impl<K, V> HashMap<K, V, RandomState>
where
    K: Key,
    V: Value,
{
    pub fn new(bits: u8) -> Self {
        Self::new_with_hasher(bits, RandomState::default())
    }
}

impl<K, V, S> HashMap<K, V, S>
where
    K: Key,
    V: Value,
    S: HashBuilder,
{
    pub fn new_with_hasher(bits: u8, hasher: S) -> Self {
        Self {
            inner: Arc::new(HashMapInner::new_with_hasher(bits, hasher)),
        }
    }

    pub fn insert(&self, key: K, value: V) -> bool {
        let guard = &epoch::pin();
        self.inner.insert(key, value, guard)
    }

    pub fn remove(&self, key: K) -> bool {
        let guard = &epoch::pin();
        self.inner.remove(key, guard)
    }

    pub fn update(&self, key: K, value: V) -> bool {
        let guard = &epoch::pin();
        let exists = self.inner.remove(key.clone(), guard);
        self.inner.insert(key, value, guard);
        exists
    }

    pub fn get(&self, key: &K) -> bool {
        let guard = epoch::pin();
        let value = self.inner.get(key, &guard);
        match value {
            Some(_) => true,
            None => false,
        }
    }
}
