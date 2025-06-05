# distutils: language = c++
# cython: language_level=3

from libcpp.string cimport string
from libcpp cimport bool

cdef extern from "parserPool.h":
    cdef cppclass ParserPool:
        ParserPool(int nReaders, int nMoveProcessors, int minSec, int maxSec, int maxInc,
                  string outdir, string elo_edges, size_t chunkSize, int printFreq,
                  size_t numThreads, int printOffset) except +
        void join()
        void enqueue(string zst, string name)
        int numCompleted()

cdef class PyParserPool:
    cdef ParserPool* _pool

    def __cinit__(self, int nReaders, int nMoveProcessors, int minSec, int maxSec, int maxInc,
                  str outdir, str elo_edges, size_t chunkSize, int printFreq,
                  size_t numThreads, int printOffset):
        self._pool = new ParserPool(nReaders, nMoveProcessors, minSec, maxSec, maxInc,
                                  outdir.encode('utf-8'), elo_edges.encode('utf-8'),
                                  chunkSize, printFreq, numThreads, printOffset)

    def __dealloc__(self):
        if self._pool != NULL:
            del self._pool
            self._pool = NULL

    def join(self):
        """Join all parser threads and close the writer."""
        if self._pool != NULL:
            self._pool.join()

    def enqueue(self, str zst, str name):
        """Enqueue a new file for parsing.
        
        Args:
            zst: Path to the zst compressed file
            name: Name identifier for the file
        """
        if self._pool != NULL:
            self._pool.enqueue(zst.encode('utf-8'), name.encode('utf-8'))
    
    def numCompleted(self):
        if self._pool != NULL:
            return self._pool.numCompleted()