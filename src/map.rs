//! The lock-free unordered list algorithm is based on Zhang, K, et al.
//! "Practical non-blocking unordered lists".

use crossbeam_epoch::{Atomic, Guard, Owned, Shared};
use std::{
    fmt::Debug,
    sync::atomic::{AtomicU8, Ordering},
};

pub trait Key: Send + Sync + PartialEq + Eq + Debug {}
pub trait Value: Send + Sync + Debug {}

impl<T: Send + Sync + PartialEq + Eq + Debug> Key for T {}
impl<T: Send + Sync + Debug> Value for T {}

const STATE_INS: u8 = 1 << 0;
const STATE_REM: u8 = 1 << 1;
const STATE_DAT: u8 = 1 << 2;
const STATE_INV: u8 = 1 << 3;

const MARK_DEL: u16 = 0x1;

#[cfg(test)]
use std::sync::atomic::AtomicUsize;
#[cfg(test)]
static CREATED: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static DESTROYED: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
static UNLINKED: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
#[repr(align(64))]
struct Node<K, V>
where
    K: Key,
    V: Value,
{
    key: K,
    value: Option<V>,
    state: AtomicU8,
    next: Atomic<Node<K, V>>,
}

impl<K, V> Node<K, V>
where
    K: Key,
    V: Value,
{
    fn new(key: K, value: Option<V>, state: u8) -> Self {
        #[cfg(test)]
        CREATED.fetch_add(1, Ordering::Relaxed);

        Self {
            key,
            value,
            state: AtomicU8::new(state),
            next: Atomic::null(),
        }
    }
}

impl<K, V> Drop for Node<K, V>
where
    K: Key,
    V: Value,
{
    fn drop(&mut self) {
        #[cfg(test)]
        DESTROYED.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Default, Debug)]
#[repr(align(64))]
pub struct ListInner<K, V>
where
    K: Key,
    V: Value,
{
    head: Atomic<Node<K, V>>,
}

impl<K, V> ListInner<K, V>
where
    K: Key,
    V: Value,
{
    pub fn insert(&self, key: K, value: V, guard: &Guard) -> bool {
        let node_ptr = Owned::new(Node::new(key, Some(value), STATE_INS)).into_shared(guard);
        let node = unsafe { node_ptr.deref() };

        self.enlist(node_ptr, guard);

        let insert = self.help_insert(node_ptr, &node.key, guard);
        if let Err(_) = node.state.compare_exchange_weak(
            STATE_INS,
            if insert { STATE_DAT } else { STATE_INV },
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            self.help_remove(node_ptr, &node.key, guard);
            node.state.store(STATE_INV, Ordering::Release);
        }
        insert
    }

    pub fn remove(&self, key: K, guard: &Guard) -> bool {
        let node_ptr = Owned::new(Node::new(key, None, STATE_REM)).into_shared(guard);
        let node = unsafe { node_ptr.deref() };

        self.enlist(node_ptr, guard);

        let remove = self.help_remove(node_ptr, &node.key, guard);
        node.state.store(STATE_INV, Ordering::Release);
        remove
    }

    pub fn get<'g>(&self, key: &'g K, guard: &'g Guard) -> Option<&'g V> {
        let curr_ptr = self.head.load(Ordering::Acquire, guard);
        let mut iter = NodeIter {
            prev_ptr: Shared::null(),
            curr_ptr,
            guard,
        };
        while iter.is_valid() {
            let curr = iter.node();
            if &curr.key == key {
                let state = curr.state.load(Ordering::Acquire);
                if state != STATE_INV {
                    if state == STATE_INS || state == STATE_DAT {
                        return curr.value.as_ref();
                    } else {
                        return None;
                    }
                }
            }
            iter.next();
        }
        None
    }

    fn enlist(&self, node_ptr: Shared<Node<K, V>>, guard: &Guard) {
        debug_assert!(!node_ptr.is_null());

        let node = unsafe { node_ptr.deref() };

        let mut head_ptr = self.head.load(Ordering::Acquire, guard);
        loop {
            node.next.store(head_ptr, Ordering::Release);
            match self.head.compare_exchange_weak(
                head_ptr,
                node_ptr,
                Ordering::AcqRel,
                Ordering::Acquire,
                guard,
            ) {
                Ok(_) => return,
                Err(e) => head_ptr = e.current,
            }
        }
    }

    fn help_insert(&self, prev_ptr: Shared<Node<K, V>>, key: &K, guard: &Guard) -> bool {
        let prev = unsafe { prev_ptr.deref() };
        let curr_ptr = prev.next.load(Ordering::Acquire, guard);
        let mut iter = NodeIter {
            prev_ptr,
            curr_ptr,
            guard,
        };

        while iter.is_valid() {
            let curr = iter.node();
            let state = curr.state.load(Ordering::Acquire);

            if state == STATE_INV {
                iter.delete();
            } else if &curr.key != key {
                iter.next();
            } else if state == STATE_REM {
                return true;
            } else if state == STATE_INS || state == STATE_DAT {
                return false;
            }
        }

        true
    }

    fn help_remove(&self, prev_ptr: Shared<Node<K, V>>, key: &K, guard: &Guard) -> bool {
        let prev = unsafe { prev_ptr.deref() };
        let curr_ptr = prev.next.load(Ordering::Acquire, guard);
        let mut iter = NodeIter {
            prev_ptr,
            curr_ptr,
            guard,
        };

        while iter.is_valid() {
            let curr = iter.node();
            let state = curr.state.load(Ordering::Acquire);

            if state == STATE_INV {
                iter.delete();
            } else if &curr.key != key {
                iter.next();
            } else if state == STATE_REM {
                return false;
            } else if state == STATE_INS {
                match curr.state.compare_exchange_weak(
                    STATE_INS,
                    STATE_REM,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return true,
                    Err(_) => {}
                }
            } else if state == STATE_DAT {
                curr.state.store(STATE_INV, Ordering::Release);
                return true;
            }
        }

        true
    }
}

#[derive(Debug)]
struct NodeIter<'g, K, V>
where
    K: Key,
    V: Value,
{
    prev_ptr: Shared<'g, Node<K, V>>,
    curr_ptr: Shared<'g, Node<K, V>>,
    guard: &'g Guard,
}

impl<'g, K, V> NodeIter<'g, K, V>
where
    K: Key,
    V: Value,
{
    fn is_valid(&self) -> bool {
        !self.curr_ptr.is_null()
    }

    fn node(&self) -> &'g Node<K, V> {
        assert!(self.is_valid());

        unsafe { self.curr_ptr.deref() }
    }

    fn next(&mut self) {
        assert!(self.is_valid());

        let node = unsafe { self.curr_ptr.deref() };
        let succ_ptr = node.next.load(Ordering::Acquire, self.guard);
        self.prev_ptr = self.curr_ptr;
        self.curr_ptr = succ_ptr;
    }

    /// Try to delete the current node. Travel next no matter if the current node is deleted,
    ///
    /// Return if this node is reclaimed by this call.
    pub fn delete(&mut self) -> bool {
        assert!(!self.curr_ptr.is_null());

        let prev = unsafe { self.prev_ptr.deref() };
        let curr = unsafe { self.curr_ptr.deref() };

        let curr_ptr = self.curr_ptr;
        let succ_ptr = curr.next.load(Ordering::Acquire, self.guard);

        // move to next node first
        self.curr_ptr = succ_ptr;

        if is_ptr_marked(curr_ptr) {
            // another thread is deleting the prev node, skip
            return false;
        }

        if is_ptr_marked(succ_ptr) {
            // another thread is deleting this node, skip
            return false;
        }

        // try to mark this ptr deleting
        if let Err(_) = curr.next.compare_exchange_weak(
            succ_ptr,
            succ_ptr.with_tag(MARK_DEL as usize),
            Ordering::AcqRel,
            Ordering::Relaxed,
            self.guard,
        ) {
            // another thread has marked this node as deleting,
            // let that thread reclaim this node
            return false;
        }

        match prev.next.compare_exchange_weak(
            curr_ptr,
            succ_ptr,
            Ordering::AcqRel,
            Ordering::Relaxed,
            self.guard,
        ) {
            Ok(_) => {
                #[cfg(test)]
                UNLINKED.fetch_add(1, Ordering::Relaxed);

                // curr node is unlinked, reclaim its memory
                let to_reclaim_ptr = curr_ptr;

                // reclaim memory by EBR
                unsafe {
                    self.guard.defer_destroy(to_reclaim_ptr);
                };

                true
            }
            Err(_) => {
                // another thread is deleting the prev node, abort
                curr.next.store(succ_ptr, Ordering::Release);
                false
            }
        }
    }
}

fn is_ptr_marked<K, V>(ptr: Shared<Node<K, V>>) -> bool
where
    K: Key,
    V: Value,
{
    ptr.tag() as u16 == MARK_DEL
}

#[cfg(test)]
mod tests {

    use std::{collections::HashSet, sync::Arc};

    use super::*;
    use crossbeam_epoch as epoch;

    #[test]
    fn test_concurrent() {
        let list = Arc::new(ListInner::default());

        let task = |i: u64, l: Arc<ListInner<u64, u64>>| {
            // std::thread::sleep(std::time::Duration::from_millis(100));

            let guard = epoch::pin();
            assert!(l.get(&i, &guard).is_none());
            drop(guard);

            let guard = epoch::pin();
            assert!(l.insert(i, i, &guard));
            drop(guard);

            let guard = epoch::pin();
            assert_eq!(l.get(&i, &guard), Some(&i));
            drop(guard);

            for _ in 0..10000 {
                let guard = epoch::pin();
                assert!(l.remove(i, &guard));
                drop(guard);

                let guard = epoch::pin();
                assert!(l.get(&i, &guard).is_none());
                drop(guard);

                let guard = epoch::pin();
                assert!(l.insert(i, i, &guard), "insert {} fail", i);
                drop(guard);

                let guard = epoch::pin();
                assert_eq!(l.get(&i, &guard), Some(&i));
                drop(guard);
            }
        };

        let mut handles = vec![];

        let threads = 16;

        let ins = std::time::Instant::now();

        let mut s = HashSet::new();
        for i in 1..(threads + 1) {
            let l = list.clone();
            s.insert(i);
            handles.push(std::thread::spawn(move || task(i, l)));
        }
        for handle in handles {
            handle.join().unwrap();
        }

        println!("ops finished in {:?}", ins.elapsed());

        assert!(!epoch::is_pinned());

        let guard = epoch::pin();

        let mut inlist = 0;

        let mut curr_ptr = list.head.load(Ordering::Acquire, &guard);
        while !curr_ptr.is_null() {
            inlist += 1;
            let curr = unsafe { curr_ptr.deref() };
            let state = curr.state.load(Ordering::Acquire);
            if state == STATE_DAT || state == STATE_INS {
                assert!(s.contains(&curr.key));
                s.remove(&curr.key);
            }
            curr_ptr = curr.next.load(Ordering::Acquire, &guard);
        }

        assert!(s.is_empty());

        drop(guard);

        assert!(!epoch::is_pinned());

        for _ in 0..1000 {
            let guard = epoch::pin();
            let f = move |guard| drop(guard);
            f(guard);
        }

        let created = CREATED.load(Ordering::Relaxed);
        let destroyed = DESTROYED.load(Ordering::Relaxed);
        let unlinked = UNLINKED.load(Ordering::Relaxed);
        let leaked = created - destroyed - inlist;
        println!(
            "created: {} destroyed: {} inlist: {} unlinked:{} inlist + unlinked:{}, leaked: {}",
            created,
            destroyed,
            inlist,
            unlinked,
            inlist + unlinked,
            leaked,
        );

        assert_eq!(inlist, threads as usize);
        assert_eq!(leaked, 0)
    }
}
