# cython: language_level=3

cdef class CZSConfig:
    cdef:
        public bint need_combine
        public str zs_combine_mode
        public bint one_bi_zs
        public str zs_algo

    def __cinit__(self, bint need_combine=True, str zs_combine_mode="zs", bint one_bi_zs=False, str zs_algo="normal"):
        self.need_combine = need_combine
        self.zs_combine_mode = zs_combine_mode
        self.one_bi_zs = one_bi_zs
        self.zs_algo = zs_algo

    def __str__(self):
        return f"CZSConfig(need_combine={self.need_combine}, zs_combine_mode={self.zs_combine_mode}, one_bi_zs={self.one_bi_zs}, zs_algo={self.zs_algo})"

    def __repr__(self):
        return self.__str__()

    cpdef CZSConfig copy(self):
        return CZSConfig(
            need_combine=self.need_combine,
            zs_combine_mode=self.zs_combine_mode,
            one_bi_zs=self.one_bi_zs,
            zs_algo=self.zs_algo
        )
