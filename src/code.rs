use std::fmt::Debug;
use std::hash::{BuildHasher, Hash};

pub trait Key: Send + Sync + PartialEq + Eq + Debug + Hash + Clone {}
pub trait Value: Send + Sync + Debug {}

impl<T: Send + Sync + PartialEq + Eq + Debug + Hash + Clone> Key for T {}
impl<T: Send + Sync + Debug> Value for T {}

pub trait HashBuilder = BuildHasher + Clone + Send + Sync + 'static;
