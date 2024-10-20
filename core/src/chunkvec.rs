use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

const CHUNK_SIZE: usize = 1024; // 每个块的大小

struct Chunk<T> {
    data: [Option<T>; CHUNK_SIZE],
}

impl<T> Chunk<T> {
    fn new() -> Self {
        Self {
            data: [(); CHUNK_SIZE].map(|_| None),
        }
    }
}

pub struct ChunkVec<T> {
    chunks: Vec<Box<Chunk<T>>>,
    len: usize,
}

impl<T> ChunkVec<T> {
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            len: 0,
        }
    }

    fn get_chunk_index_and_offset(&self, index: usize) -> (usize, usize) {
        let chunk_index = index / CHUNK_SIZE;
        let offset = index % CHUNK_SIZE;
        (chunk_index, offset)
    }

    pub fn push(&mut self, value: T) {
        let (chunk_index, offset) = self.get_chunk_index_and_offset(self.len);
        if chunk_index >= self.chunks.len() {
            self.chunks.push(Box::new(Chunk::new()));
        }
        self.chunks[chunk_index].data[offset] = Some(value);
        self.len += 1;
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        let (chunk_index, offset) = self.get_chunk_index_and_offset(self.len);
        self.chunks[chunk_index].data[offset].take()
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        let (chunk_index, offset) = self.get_chunk_index_and_offset(index);
        self.chunks[chunk_index].data[offset].as_ref()
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        let (chunk_index, offset) = self.get_chunk_index_and_offset(index);
        self.chunks[chunk_index].data[offset].as_mut()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
        self.len = 0;
    }
}

impl<T> Deref for ChunkVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe {
            let slice = std::slice::from_raw_parts(self.as_ptr(), self.len);
            slice
        }
    }
}

impl<T> DerefMut for ChunkVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            let slice = std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len);
            slice
        }
    }
}

impl<T> ChunkVec<T> {
    fn as_ptr(&self) -> *const T {
        if self.len == 0 {
            return std::ptr::null();
        }
        let (chunk_index, offset) = self.get_chunk_index_and_offset(0);
        self.chunks[chunk_index].data[offset].as_ref().unwrap() as *const T
    }

    fn as_mut_ptr(&mut self) -> *mut T {
        if self.len == 0 {
            return std::ptr::null_mut();
        }
        let (chunk_index, offset) = self.get_chunk_index_and_offset(0);
        self.chunks[chunk_index].data[offset].as_mut().unwrap() as *mut T
    }
}

impl<T> Drop for ChunkVec<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

// 实现迭代器
pub struct ChunkVecIterator<'a, T> {
    chunk_vec: &'a ChunkVec<T>,
    index: usize,
}

impl<'a, T> Iterator for ChunkVecIterator<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.chunk_vec.len() {
            return None;
        }
        let (chunk_index, offset) = self.chunk_vec.get_chunk_index_and_offset(self.index);
        self.index += 1;
        self.chunk_vec.chunks[chunk_index].data[offset].as_ref()
    }
}

impl<'a, T> IntoIterator for &'a ChunkVec<T> {
    type Item = &'a T;
    type IntoIter = ChunkVecIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        ChunkVecIterator {
            chunk_vec: self,
            index: 0,
        }
    }
}

// 实现可变迭代器
pub struct ChunkVecMutIterator<'a, T> {
    chunk_vec: &'a mut ChunkVec<T>,
    index: usize,
}

impl<'a, T> Iterator for ChunkVecMutIterator<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.chunk_vec.len() {
            return None;
        }
        let (chunk_index, offset) = self.chunk_vec.get_chunk_index_and_offset(self.index);
        self.index += 1;
        let chunk = &mut self.chunk_vec.chunks[chunk_index];
        let ptr = chunk.data[offset].as_mut().unwrap() as *mut T;
        unsafe { Some(&mut *ptr) }
    }
}

impl<'a, T> IntoIterator for &'a mut ChunkVec<T> {
    type Item = &'a mut T;
    type IntoIter = ChunkVecMutIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        ChunkVecMutIterator {
            chunk_vec: self,
            index: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop() {
        let mut vec = ChunkVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.pop(), Some(2));
        assert_eq!(vec.pop(), Some(1));
        assert_eq!(vec.pop(), None);
    }

    #[test]
    fn test_get() {
        let mut vec = ChunkVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert_eq!(vec.get(0), Some(&1));
        assert_eq!(vec.get(1), Some(&2));
        assert_eq!(vec.get(2), Some(&3));
        assert_eq!(vec.get(3), None);
    }

    #[test]
    fn test_get_mut() {
        let mut vec = ChunkVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        *vec.get_mut(1).unwrap() = 4;
        assert_eq!(vec.get(1), Some(&4));
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut vec = ChunkVec::new();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
        vec.push(1);
        assert_eq!(vec.len(), 1);
        assert!(!vec.is_empty());
        vec.pop();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_clear() {
        let mut vec = ChunkVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        vec.clear();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn test_iterator() {
        let mut vec = ChunkVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        let mut iter = vec.into_iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_mut_iterator() {
        let mut vec = ChunkVec::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        for item in &mut vec {
            *item += 1;
        }
        let mut iter = vec.into_iter();
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);
    }
}
